"""sparkrun tune group and subcommands."""

from __future__ import annotations

import logging
import sys

import click

from sparkrun.core.application_profile import render_identity_text
from sparkrun.core.features import feature_gate_enabled

from ._common import (
    RECIPE_NAME,
    _get_context,
    _load_recipe,
    dry_run_option,
    host_options,
    resolve_cluster_config,
    with_host_context,
)

logger = logging.getLogger(__name__)


def _require_runtime_family(recipe, v, family: str, requirement: str):
    """Resolve *recipe*'s runtime plugin, requiring it to be in *family*.

    Matching on the runtime family rather than a hardcoded name list means a
    new variant — ``vllm-ray``, ``eugr-vllm``, or an out-of-tree one — is
    tunable without editing this module.  An unregistered runtime is, by
    definition, not in the family, so it gets the same message as a wrong
    one rather than a traceback out of ``get_runtime``.

    Exits with an error message when the recipe doesn't qualify.
    """
    from sparkrun.core.bootstrap import get_runtime

    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError:
        runtime = None

    if runtime is None or runtime.get_family() != family:
        click.echo("Error: %s (got runtime=%r)" % (requirement, recipe.runtime), err=True)
        sys.exit(1)
    return runtime


@click.group(hidden=not feature_gate_enabled("cli.tune"))
@click.pass_context
def tune(ctx):
    """Tune runtime kernels for optimal performance."""
    if not _get_context(ctx).config.is_feature_enabled("cli.tune"):
        raise click.ClickException(render_identity_text("Tuning is disabled. Enable it with: {app_command} setup features enable cli.tune"))


@tune.command("sglang")
@click.argument("recipe_name", type=RECIPE_NAME)
@host_options
@click.option("--tp", "tp_sizes", type=int, multiple=True, help="TP size(s) to tune (repeatable; default: 1,2,4,8)")
@click.option("--image", default=None, help="Override container image")
@click.option("--output-dir", default=None, help="Override tuning config output directory")
@click.option("--skip-clone", is_flag=True, help="Skip cloning SGLang repo (scripts already in image)")
@click.option(
    "--mode", type=click.Choice(["moe", "fp8"]), default="moe", show_default=True, help="Tune fused MoE or dense block-FP8 kernels"
)
@click.option(
    "--shape",
    "shapes",
    type=(click.IntRange(min=1), click.IntRange(min=1)),
    multiple=True,
    help="Per-rank FP8 weight shape: N K (repeatable; required for --mode fp8)",
)
@click.option(
    "--block-shape",
    type=(click.IntRange(min=16), click.IntRange(min=16)),
    default=(128, 128),
    help="FP8 quantization block shape: N K (e.g. 32 32)",
)
@click.option("--batch-size", "batch_sizes", type=click.IntRange(min=1), multiple=True, help="Token count M to tune (repeatable; FP8 only)")
@click.option("--parallel", "-j", type=int, default=1, help="Run N tuning jobs concurrently (default: 1 = sequential)")
@click.option(
    "--timeout",
    type=click.IntRange(min=0),
    default=0,
    help="Per-TP tuning timeout in seconds (default: 0 = no timeout)",
)
@dry_run_option
@click.pass_context
@with_host_context
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
    mode,
    shapes,
    block_shape,
    batch_sizes,
    parallel,
    timeout,
    dry_run,
    config_path=None,
    *,
    host_list: list[str],
    cluster_mgr=None,
):
    """Tune SGLang fused MoE Triton kernels or dense block-FP8 GEMMs.

    Runs Triton kernel autotuning inside the recipe's container on a single
    host.  Generates optimal tile configs (BLOCK_M/N/K, warps, stages) for
    each TP size and saves them for automatic use in future inference runs.

    RECIPE_NAME provides the model name and container image.

    \b
    Examples:
      {app_command} tune sglang qwen3.5-35b-bf16-sglang -H 192.168.11.13
      {app_command} tune sglang qwen3.5-35b-bf16-sglang --cluster mylab --tp 4
      {app_command} tune sglang qwen3.5-35b-bf16-sglang -H myhost --tp 1 --tp 2 --tp 4
      {app_command} tune sglang qwen3.5-35b-bf16-sglang -H myhost --parallel 2
    """
    from sparkrun.tuning.sglang import SglangTuner, DEFAULT_TP_SIZES

    if mode == "fp8" and not shapes:
        raise click.UsageError("--mode fp8 requires --shape N K (use the per-rank shapes reported by the runtime)")
    if mode != "fp8" and (shapes or batch_sizes or block_shape != (128, 128)):
        raise click.UsageError("--shape, --block-shape, and --batch-size require --mode fp8")
    if any(n & (n - 1) for n in block_shape):
        raise click.UsageError("--block-shape values must be powers of two")
    if mode == "fp8" and (len(tp_sizes) > 1 or parallel != 1):
        raise click.UsageError("FP8 shapes are already per-rank: use one --tp value and --parallel 1")

    sctx = _get_context(ctx)
    v = sctx.variables
    config = sctx.config

    recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name)

    runtime = _require_runtime_family(recipe, v, "sglang", "tune sglang requires an SGLang recipe")

    # host_list injected by @with_host_context; only use the first host
    target_host = host_list[0]
    if len(host_list) > 1:
        logger.info("Tuning runs on a single host; using first host: %s", target_host)

    # Resolve remote cache dir.  When the cluster doesn't define an explicit
    # cache_dir, probe the target so the path reflects the SSH login user's
    # $HOME / HF_HOME instead of the control machine's.
    from sparkrun.core.launcher import resolve_effective_cache_dir
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    cluster_cfg = resolve_cluster_config(cluster_name, hosts, hosts_file, cluster_mgr)
    from sparkrun.core.images import ImagePlanError, resolve_runtime_image_plan

    image_cluster = cluster_mgr.get(cluster_cfg.name) if cluster_cfg.name and cluster_mgr else None
    try:
        container_image = image or resolve_runtime_image_plan(recipe, runtime, [target_host], cluster=image_cluster).head_image()
    except ImagePlanError as error:
        raise click.ClickException(str(error)) from error
    remote_cache_dir = resolve_effective_cache_dir(
        cluster_cfg.cache_dir,
        host_list,
        build_ssh_kwargs(config),
        config,
        dry_run=dry_run,
    )

    # Default TP sizes
    effective_tp = tuple(tp_sizes) if tp_sizes else (int(recipe.defaults.get("tensor_parallel", 1)),) if mode == "fp8" else DEFAULT_TP_SIZES

    tuner = SglangTuner(
        host=target_host,
        image=container_image,
        model=recipe.model,
        config=config,
        cache_dir=remote_cache_dir,
        output_dir=output_dir,
        skip_clone=skip_clone,
        timeout=timeout,
        dry_run=dry_run,
        mode=mode,
        shapes=shapes,
        block_shape=block_shape,
        batch_sizes=batch_sizes,
    )

    rc = tuner.run_tuning(tp_sizes=effective_tp, parallel=parallel)
    sys.exit(rc)


@tune.command("vllm")
@click.argument("recipe_name", type=RECIPE_NAME)
@host_options
@click.option("--tp", "tp_sizes", type=int, multiple=True, help="TP size(s) to tune (repeatable; default: 1,2,4,8)")
@click.option("--image", default=None, help="Override container image")
@click.option("--output-dir", default=None, help="Override tuning config output directory")
@click.option(
    "--mode",
    type=click.Choice(["moe", "fp8", "all"]),
    default="all",
    show_default=True,
    help="Kernels to tune: moe (fused MoE), fp8 (dense GEMM), or all",
)
@click.option(
    "--vllm-tune-ref",
    default=None,
    help="Override the vllm-tune git ref (tag/branch/SHA) pinned in config",
)
@click.option("--parallel", "-j", type=int, default=1, help="Run N tuning jobs concurrently (default: 1 = sequential)")
@click.option(
    "--timeout",
    type=click.IntRange(min=0),
    default=0,
    help="Per-TP tuning timeout in seconds (default: 0 = no timeout)",
)
@dry_run_option
@click.pass_context
@with_host_context
def tune_vllm(
    ctx,
    recipe_name,
    hosts,
    hosts_file,
    cluster_name,
    tp_sizes,
    image,
    output_dir,
    mode,
    vllm_tune_ref,
    parallel,
    timeout,
    dry_run,
    config_path=None,
    *,
    host_list: list[str],
    cluster_mgr=None,
):
    """Tune vLLM Triton kernels for DGX Spark via the vllm-tune backend.

    Shells out to https://github.com/SeraphimSerapis/vllm-tune on the target
    host.  Tunes both fused MoE kernels and FP8 dense GEMM kernels by default
    (override with ``--mode``).  Produced configs land in the same flat cache
    directory that vLLM runtimes auto-mount, so subsequent ``{app_command} run``
    invocations pick them up without further action.

    RECIPE_NAME provides the model name and container image.  vllm-tune
    launches its own standalone tuning container per TP size, so a running
    inference workload is not required.

    Prerequisites on the remote host: ``jq``, ``docker``, ``git``.

    \b
    Examples:
      {app_command} tune vllm qwen3-moe-vllm -H 192.168.11.13
      {app_command} tune vllm qwen3-moe-vllm --cluster mylab --tp 4 --mode moe
      {app_command} tune vllm qwen3-4b-fp8 -H myhost --mode fp8 --tp 1
      {app_command} tune vllm qwen3-moe-vllm -H myhost --tp 1 --tp 2 --tp 4
      {app_command} tune vllm qwen3-moe-vllm -H myhost --parallel 2
    """
    from sparkrun.tuning.vllm import VllmTuner, DEFAULT_TP_SIZES

    sctx = _get_context(ctx)
    v = sctx.variables
    config = sctx.config

    recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name)

    runtime = _require_runtime_family(recipe, v, "vllm", "tune vllm requires a vLLM recipe")

    # host_list injected by @with_host_context; only use the first host
    target_host = host_list[0]
    if len(host_list) > 1:
        logger.info("Tuning runs on a single host; using first host: %s", target_host)

    # Resolve remote cache dir.  When the cluster doesn't define an explicit
    # cache_dir, probe the target so the path reflects the SSH login user's
    # $HOME / HF_HOME instead of the control machine's.
    from sparkrun.core.launcher import resolve_effective_cache_dir
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    cluster_cfg = resolve_cluster_config(cluster_name, hosts, hosts_file, cluster_mgr)
    from sparkrun.core.images import ImagePlanError, resolve_runtime_image_plan

    image_cluster = cluster_mgr.get(cluster_cfg.name) if cluster_cfg.name and cluster_mgr else None
    try:
        container_image = image or resolve_runtime_image_plan(recipe, runtime, [target_host], cluster=image_cluster).head_image()
    except ImagePlanError as error:
        raise click.ClickException(str(error)) from error
    remote_cache_dir = resolve_effective_cache_dir(
        cluster_cfg.cache_dir,
        host_list,
        build_ssh_kwargs(config),
        config,
        dry_run=dry_run,
    )

    # Default TP sizes
    effective_tp = tuple(tp_sizes) if tp_sizes else DEFAULT_TP_SIZES

    tuner = VllmTuner(
        host=target_host,
        image=container_image,
        model=recipe.model,
        config=config,
        cache_dir=remote_cache_dir,
        output_dir=output_dir,
        mode=mode,
        vllm_tune_ref=vllm_tune_ref,
        dry_run=dry_run,
        timeout=timeout,
    )

    rc = tuner.run_tuning(tp_sizes=effective_tp, parallel=parallel)
    sys.exit(rc)
