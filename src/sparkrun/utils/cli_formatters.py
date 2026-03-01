"""Presentation layer formatting functions for sparkrun CLI."""

from typing import Any

import click


def format_recipe_table(
        recipes: list[dict[str, Any]],
        *,
        show_model: bool = False,
        show_file: bool = False,
) -> str:
    """Format recipe metadata as a text table.

    Args:
        recipes: Recipe metadata dicts.
        show_model: Include a Model column (used by search results).
        show_file: Include a File column (used by list output).

    Returns:
        Formatted multi-line string (no trailing newline).
    """
    if not recipes:
        return "No recipes found."

    # Pre-compute display values
    reg_names = [r.get("registry", "local") for r in recipes]
    tp_vals = [str(r["tp"]) if r.get("tp", "") != "" else "-" for r in recipes]
    mn_vals = [str(r.get("min_nodes", 1)) for r in recipes]
    gm_vals = [str(r["gpu_mem"]) if r.get("gpu_mem", "") != "" else "-" for r in recipes]

    # Column widths
    w_name = max(len("Name"), *(len(r["name"]) for r in recipes)) + 2
    w_rt = max(len("Runtime"), *(len(r.get("runtime", "")) for r in recipes)) + 2
    w_tp = max(len("TP"), *(len(v) for v in tp_vals)) + 2
    w_mn = max(len("Nodes"), *(len(v) for v in mn_vals)) + 2
    w_gm = max(len("GPU Mem"), *(len(v) for v in gm_vals)) + 2
    w_reg = max(len("Registry"), *(len(n) for n in reg_names)) + 2

    # Build column definitions: (header, width, values)
    columns: list[tuple[str, int, list[str]]] = [
        ("Name", w_name, [r["name"] for r in recipes]),
        ("Runtime", w_rt, [r.get("runtime", "") for r in recipes]),
        ("TP", w_tp, tp_vals),
        ("Nodes", w_mn, mn_vals),
        ("GPU Mem", w_gm, gm_vals),
    ]

    if show_model:
        models = [r.get("model", "") for r in recipes]
        w_model = max(len("Model"), *(len(m) for m in models)) + 2
        columns.append(("Model", w_model, models))

    columns.append(("Registry", w_reg, reg_names))

    if show_file:
        w_file = max(len("File"), *(len(r["file"]) for r in recipes))
        columns.append(("File", w_file, [r["file"] for r in recipes]))

    # Render
    header = " ".join(f"{col[0]:<{col[1]}}" for col in columns)
    total_width = sum(col[1] for col in columns) + len(columns) - 1
    separator = "-" * total_width

    lines = [header, separator]
    for i in range(len(recipes)):
        row = " ".join(f"{col[2][i]:<{col[1]}}" for col in columns)
        lines.append(row)

    return "\n".join(lines)


def format_job_label(meta: dict[str, Any], cluster_id: str) -> str:
    """Format a display label from job metadata."""
    label = meta.get("recipe", cluster_id)
    tp = meta.get("tensor_parallel")
    if tp:
        label += f"  (tp={tp})"
    return label


def format_job_commands(meta: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (logs_cmd, stop_cmd) strings from cached metadata."""
    # Prefer recipe_ref (@spark-arena/UUID or URL) over bare recipe name
    recipe_name = meta.get("recipe_ref") or meta.get("recipe")
    if not recipe_name:
        return None, None
    job_hosts = meta.get("hosts", [])
    tp = meta.get("tensor_parallel")
    host_flag = f" --hosts {','.join(job_hosts)}" if job_hosts else ""
    tp_flag = f" --tp {tp}" if tp else ""
    logs_cmd = f"sparkrun logs {recipe_name}{host_flag}{tp_flag}"
    stop_cmd = f"sparkrun stop {recipe_name}{host_flag}{tp_flag}"
    return logs_cmd, stop_cmd


def format_host_display(host: str, meta: dict[str, Any] | None) -> str:
    """Format host with complementary IP from job metadata if available.

    If the queried host matches a mgmt or IB IP in the metadata,
    show the other IP in parentheses so both are visible.
    """
    mgmt_map = meta.get("mgmt_ip_map", {}) if meta else {}
    ib_map = meta.get("ib_ip_map", {}) if meta else {}
    mgmt = mgmt_map.get(host)
    if mgmt and mgmt != host:
        return f"{host} (mgmt: {mgmt})"
    ib = ib_map.get(host)
    if ib and ib != host:
        return f"{host} (ib: {ib})"
    return host


def display_recipe_detail(recipe, show_vram=True, registry_name=None, cli_overrides=None, cache_dir=None):
    """Display recipe details (shared by show and recipe show commands)."""
    click.echo(f"Name:         {recipe.name}")
    click.echo(f"Description:  {recipe.description}")
    if recipe.maintainer:
        click.echo(f"Maintainer:   {recipe.maintainer}")
    click.echo(f"Runtime:      {recipe.runtime}")
    click.echo(f"Model:        {recipe.model}")
    click.echo(f"Container:    {recipe.container}")
    max_nodes = recipe.max_nodes or "unlimited"
    click.echo(f"Nodes:        {recipe.min_nodes} - {max_nodes}")
    click.echo(f"Repository:   {registry_name or 'Local'}")
    click.echo(f"File Path:    {recipe.source_path}")

    if recipe.defaults:
        click.echo("\nDefaults:")
        for k, v in sorted(recipe.defaults.items()):
            click.echo(f"  {k}: {v}")

    if recipe.env:
        click.echo("\nEnvironment:")
        for k, v in sorted(recipe.env.items()):
            click.echo(f"  {k}={v}")

    if recipe.command:
        click.echo(f"\nCommand:\n  {recipe.command.strip()}")

    if show_vram:
        display_vram_estimate(recipe, cli_overrides=cli_overrides, cache_dir=cache_dir)


def display_vram_estimate(recipe, cli_overrides=None, auto_detect=True, cache_dir=None):
    """Display VRAM estimation for a recipe."""
    from sparkrun.models.vram import DGX_SPARK_VRAM_GB

    try:
        est = recipe.estimate_vram(cli_overrides=cli_overrides, auto_detect=auto_detect, cache_dir=cache_dir)
    except Exception as e:
        click.echo(f"\nVRAM estimation failed: {e}", err=True)
        return

    click.echo("\nVRAM Estimation:")
    if est.model_dtype:
        click.echo(f"  Model dtype:      {est.model_dtype}")
    if est.model_params:
        click.echo(f"  Model params:     {est.model_params:,}")
    click.echo(f"  KV cache dtype:   {est.kv_dtype or 'bfloat16 (default)'}")
    if all([est.num_layers, est.num_kv_heads, est.head_dim]):
        click.echo(f"  Architecture:     {est.num_layers} layers, {est.num_kv_heads} KV heads, {est.head_dim} head_dim")
    click.echo(f"  Model weights:    {est.model_weights_gb:.2f} GB")
    if est.kv_cache_total_gb is not None:
        click.echo(f"  KV cache:         {est.kv_cache_total_gb:.2f} GB (max_model_len={est.max_model_len:,})")
    click.echo(f"  Tensor parallel:  {est.tensor_parallel}")
    click.echo(f"  Per-GPU total:    {est.total_per_gpu_gb:.2f} GB")
    fit_str = "YES" if est.fits_dgx_spark else "EXCEEDS %.0f GB" % DGX_SPARK_VRAM_GB
    click.echo(f"  DGX Spark fit:    {fit_str}")

    # GPU memory budget analysis
    if est.gpu_memory_utilization is not None:
        click.echo("\n  GPU Memory Budget:")
        click.echo(f"    gpu_memory_utilization: {est.gpu_memory_utilization:.0%}")
        click.echo(f"    Usable GPU memory:     {est.usable_gpu_memory_gb:.1f} GB"
                   f" ({DGX_SPARK_VRAM_GB:.0f} GB x {est.gpu_memory_utilization:.0%})")
        click.echo(f"    Available for KV:      {est.available_kv_gb:.1f} GB")
        if est.max_context_tokens is not None:
            click.echo(f"    Max context tokens:    {est.max_context_tokens:,}")
            if est.context_multiplier is not None and est.max_model_len:
                click.echo(f"    Context multiplier:    {est.context_multiplier:.1f}x"
                           f" (vs max_model_len={est.max_model_len:,})")
                if est.context_multiplier < 1.0:
                    click.echo(f"    WARNING: max_model_len exceeds available KV budget"
                               f" ({est.context_multiplier:.1%} fits)")

    for w in est.warnings:
        click.echo(f"  Warning: {w}")
