"""sparkrun tune group and subcommands."""

from __future__ import annotations

import logging
import sys

import click

from ._common import (
    RECIPE_NAME,
    _get_context,
    _load_recipe,
    _resolve_hosts_or_exit,
    dry_run_option,
    host_options,
    resolve_cluster_config,
)

logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def tune(ctx):
    """Tune runtime kernels for optimal performance."""
    pass


@tune.command("sglang")
@click.argument("recipe_name", type=RECIPE_NAME)
@host_options
@click.option("--tp", "tp_sizes", type=int, multiple=True, help="TP size(s) to tune (repeatable; default: 1,2,4,8)")
@click.option("--image", default=None, help="Override container image")
@click.option("--output-dir", default=None, help="Override tuning config output directory")
@click.option("--skip-clone", is_flag=True, help="Skip cloning SGLang repo (scripts already in image)")
@click.option("--parallel", "-j", type=int, default=1, help="Run N tuning jobs concurrently (default: 1 = sequential)")
@dry_run_option
@click.pass_context
def tune_sglang(
    ctx,
    recipe_name,
    hosts,
    hosts_file,
    cluster_name,
    tp_sizes,
    image,
    output_dir,
    skip_clone,
    parallel,
    dry_run,
    config_path=None,
):
    """Tune SGLang fused MoE Triton kernels for DGX Spark.

    Runs Triton kernel autotuning inside the recipe's container on a single
    host.  Generates optimal tile configs (BLOCK_M/N/K, warps, stages) for
    each TP size and saves them for automatic use in future inference runs.

    RECIPE_NAME provides the model name and container image.

    \b
    Examples:
      sparkrun tune sglang qwen3.5-35b-bf16-sglang -H 192.168.11.13
      sparkrun tune sglang qwen3.5-35b-bf16-sglang --cluster mylab --tp 4
      sparkrun tune sglang qwen3.5-35b-bf16-sglang -H myhost --tp 1 --tp 2 --tp 4
      sparkrun tune sglang qwen3.5-35b-bf16-sglang -H myhost --parallel 2
    """
    from sparkrun.core.bootstrap import get_runtime
    from sparkrun.tuning.sglang import SglangTuner, DEFAULT_TP_SIZES

    sctx = _get_context(ctx)
    v = sctx.variables
    config = sctx.config

    recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name)

    # Validate runtime is sglang
    if recipe.runtime != "sglang":
        click.echo(
            "Error: tune sglang requires an SGLang recipe (got runtime=%r)" % recipe.runtime,
            err=True,
        )
        sys.exit(1)

    # Resolve container image
    runtime = get_runtime(recipe.runtime, v)
    container_image = image or runtime.resolve_container(recipe)

    # Resolve hosts — only use the first host
    host_list, _cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, sctx=sctx)
    target_host = host_list[0]
    if len(host_list) > 1:
        logger.info("Tuning runs on a single host; using first host: %s", target_host)

    # Resolve remote cache dir from cluster config
    cluster_cfg = resolve_cluster_config(cluster_name, hosts, hosts_file, _cluster_mgr)
    remote_cache_dir = cluster_cfg.cache_dir  # None is fine — tuner/build_volumes handles default

    # Default TP sizes
    effective_tp = tuple(tp_sizes) if tp_sizes else DEFAULT_TP_SIZES

    tuner = SglangTuner(
        host=target_host,
        image=container_image,
        model=recipe.model,
        config=config,
        cache_dir=remote_cache_dir,
        output_dir=output_dir,
        skip_clone=skip_clone,
        dry_run=dry_run,
    )

    rc = tuner.run_tuning(tp_sizes=effective_tp, parallel=parallel)
    sys.exit(rc)


VLLM_RUNTIMES = {"vllm-ray", "vllm-distributed", "eugr-vllm"}


@tune.command("vllm")
@click.argument("recipe_name", type=RECIPE_NAME)
@host_options
@click.option("--tp", "tp_sizes", type=int, multiple=True, help="TP size(s) to tune (repeatable; default: 1,2,4,8)")
@click.option("--image", default=None, help="Override container image")
@click.option("--output-dir", default=None, help="Override tuning config output directory")
@click.option("--skip-clone", is_flag=True, help="Skip cloning vLLM repo (scripts already in image)")
@click.option("--parallel", "-j", type=int, default=1, help="Run N tuning jobs concurrently (default: 1 = sequential)")
@dry_run_option
@click.pass_context
def tune_vllm(
    ctx,
    recipe_name,
    hosts,
    hosts_file,
    cluster_name,
    tp_sizes,
    image,
    output_dir,
    skip_clone,
    parallel,
    dry_run,
    config_path=None,
):
    """Tune vLLM fused MoE Triton kernels for DGX Spark.

    Runs Triton kernel autotuning inside the recipe's container on a single
    host.  Generates optimal tile configs (BLOCK_M/N/K, warps, stages) for
    each TP size and saves them for automatic use in future inference runs.

    RECIPE_NAME provides the model name and container image.

    \b
    Examples:
      sparkrun tune vllm qwen3-moe-vllm -H 192.168.11.13
      sparkrun tune vllm qwen3-moe-vllm --cluster mylab --tp 4
      sparkrun tune vllm qwen3-moe-vllm -H myhost --tp 1 --tp 2 --tp 4
      sparkrun tune vllm qwen3-moe-vllm -H myhost --parallel 2
    """
    from sparkrun.core.bootstrap import get_runtime
    from sparkrun.tuning.vllm import VllmTuner, DEFAULT_TP_SIZES

    sctx = _get_context(ctx)
    v = sctx.variables
    config = sctx.config

    recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name)

    # Validate runtime is a vLLM variant
    if recipe.runtime not in VLLM_RUNTIMES:
        click.echo(
            "Error: tune vllm requires a vLLM recipe (got runtime=%r)" % recipe.runtime,
            err=True,
        )
        sys.exit(1)

    # Resolve container image
    runtime = get_runtime(recipe.runtime, v)
    container_image = image or runtime.resolve_container(recipe)

    # Resolve hosts — only use the first host
    host_list, _cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, sctx=sctx)
    target_host = host_list[0]
    if len(host_list) > 1:
        logger.info("Tuning runs on a single host; using first host: %s", target_host)

    # Resolve remote cache dir from cluster config
    cluster_cfg = resolve_cluster_config(cluster_name, hosts, hosts_file, _cluster_mgr)
    remote_cache_dir = cluster_cfg.cache_dir  # None is fine — tuner/build_volumes handles default

    # Default TP sizes
    effective_tp = tuple(tp_sizes) if tp_sizes else DEFAULT_TP_SIZES

    tuner = VllmTuner(
        host=target_host,
        image=container_image,
        model=recipe.model,
        config=config,
        cache_dir=remote_cache_dir,
        output_dir=output_dir,
        skip_clone=skip_clone,
        dry_run=dry_run,
    )

    rc = tuner.run_tuning(tp_sizes=effective_tp, parallel=parallel)
    sys.exit(rc)
