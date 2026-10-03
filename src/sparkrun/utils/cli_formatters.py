"""Presentation layer formatting functions for sparkrun CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import click

from sparkrun.utils.runtime_display import RUNTIME_DISPLAY  # noqa: F401 — presentation export

if TYPE_CHECKING:
    from collections.abc import Container

    from sparkrun.core.monitoring import HostMonitorState
    from sparkrun.orchestration.disk_info import CacheStatus


def format_cache_status_table(
    host_status: dict[str, "CacheStatus"],
    local_status: "CacheStatus | None" = None,
    highlight_hosts: list[str] | set[str] | None = None,
) -> str:
    """Render the cache-status table used by ``cluster inspect``.

    Layout matches the existing ``cluster inspect`` output so the same
    eyes that have learned to read it work for the error-path display
    in :func:`sparkrun.models.distribute.distribute_model_from_local`.

    Args:
        host_status: ``host → CacheStatus`` from
            :func:`sparkrun.orchestration.disk_info.probe_cache_status`.
        local_status: Optional ``(local)`` row from
            :func:`sparkrun.orchestration.disk_info.probe_local_cache_status`.
            Rendered first when present.
        highlight_hosts: Hosts to mark with a leading ``→`` (typically
            the ones that ran out of space).
    """
    if not host_status and local_status is None:
        return ""

    highlight = set(highlight_hosts or ())

    # Width 30 matches the legacy cluster_inspect convention so the
    # output is visually identical to what users already know.
    row_fmt = "  %-2s %-30s %-10s %-10s %-10s %-10s %-12s %s"
    lines: list[str] = []
    lines.append(row_fmt % ("", "Host", "SR exists", "SR size", "HF exists", "HF size", "Free Space", "HF path"))
    lines.append("  " + "-" * 112)

    def _emit(status: "CacheStatus", marker: str) -> None:
        if status.error:
            lines.append("  %-2s %-30s Error: %s" % (marker, status.host, status.error))
            return
        lines.append(
            row_fmt
            % (
                marker,
                status.host,
                "yes" if status.sparkrun_exists else "no",
                status.sparkrun_size,
                "yes" if status.hf_exists else "no",
                status.hf_size,
                status.free_space,
                status.hf_dir,
            )
        )

    if local_status is not None:
        _emit(local_status, "  " if local_status.host not in highlight else "→ ")

    for host in sorted(host_status):
        marker = "→ " if host in highlight else "  "
        _emit(host_status[host], marker)

    return "\n".join(lines)


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
    short_id = cluster_id.removeprefix("sparkrun_")  # [:8]
    label = meta.get("recipe", cluster_id)
    tp = meta.get("tensor_parallel")
    pp = meta.get("pipeline_parallel")
    if tp or pp:
        parts = []
        if tp:
            parts.append("tp=%s" % tp)
        if pp:
            parts.append("pp=%s" % pp)
        label += "  (%s)" % ", ".join(parts)
    label += f"  [{short_id}]"
    return label


def format_job_commands(meta: dict[str, Any], cluster_id: str | None = None) -> tuple[str | None, str | None]:
    """Return (logs_cmd, stop_cmd) strings.

    When *cluster_id* is provided, emits short cluster-ID-based commands
    that are always unambiguous.  Falls back to recipe-name-based commands
    (with host/tp/port flags) when no cluster_id is available.
    """
    if cluster_id:
        short_id = cluster_id.removeprefix("sparkrun_")
        return f"sparkrun logs {short_id}", f"sparkrun stop {short_id}"
    # Fallback: recipe-based (for jobs without metadata)
    recipe_name = meta.get("recipe_ref") or meta.get("recipe")
    if not recipe_name:
        return None, None
    job_hosts = meta.get("hosts", [])
    tp = meta.get("tensor_parallel")
    port = meta.get("port")
    served_name = meta.get("served_model_name")
    host_flag = f" --hosts {','.join(job_hosts)}" if job_hosts else ""
    tp_flag = f" --tp {tp}" if tp else ""
    port_flag = f" --port {port}" if port else ""
    name_flag = f" --served-model-name {served_name}" if served_name else ""
    logs_cmd = f"sparkrun logs {recipe_name}{host_flag}{tp_flag}{port_flag}{name_flag}"
    stop_cmd = f"sparkrun stop {recipe_name}{host_flag}{tp_flag}{port_flag}{name_flag}"
    return logs_cmd, stop_cmd


def format_elapsed(seconds: float) -> str:
    """Format an elapsed duration as ``4m47s`` / ``12s``."""
    mins, secs = divmod(int(seconds or 0), 60)
    return "%dm%02ds" % (mins, secs) if mins else "%ds" % secs


def format_pending_op(op: dict[str, Any], *, with_detail: bool = True) -> str:
    """One-line summary of a pending-operation lock.

    ``<label>: <operation> (<elapsed>)`` plus the model or image the
    operation is moving, whichever the operation names.  *with_detail*
    ``False`` drops that suffix, for per-host lines where the same op is
    already spelled out in full elsewhere.
    """
    label = op.get("recipe") or op.get("cluster") or op.get("cluster_id", "?")
    detail = str(op.get("operation", "unknown")).replace("_", " ")
    line = "%s: %s (%s)" % (label, detail, format_elapsed(op.get("elapsed_seconds", 0)))
    if not with_detail:
        return line
    if op.get("model") and "model" in detail:
        return "%s  model=%s" % (line, op["model"])
    if op.get("image") and "image" in detail:
        return "%s  image=%s" % (line, op["image"])
    return line


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
    click.echo(f"Name:         {recipe.qualified_name}")
    click.echo(f"Description:  {recipe.description}")
    if recipe.maintainer:
        click.echo(f"Maintainer:   {recipe.maintainer}")
    spark_arena_benchmarks = recipe.metadata.get("spark_arena_benchmarks", [])
    if len(spark_arena_benchmarks) == 1:
        click.echo("Spark Arena:  https://spark-arena.com/benchmarks/%s" % spark_arena_benchmarks[0]["uuid"])
    elif len(spark_arena_benchmarks) > 1:
        click.echo("Spark Arena:")
        for entry in spark_arena_benchmarks:
            click.echo("  tp%s: https://spark-arena.com/benchmarks/%s" % (entry["tp"], entry["uuid"]))
    click.echo(f"Runtime:      {recipe.runtime}")
    click.echo(f"Model:        {recipe.model}")
    click.echo(f"Container:    {recipe.container}")
    max_nodes = recipe.max_nodes or "unlimited"
    click.echo(f"Nodes:        {recipe.min_nodes} - {max_nodes}")
    # click.echo(f"Registry:     {registry_name or 'N/A'}")
    # click.echo(f"File Path:    {recipe.source_path}")

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


def _resolve_target_accelerator(cluster, placement):
    """Smallest assigned capacity; unknown if any selected device is unsized."""
    if cluster is None:
        return None, None
    from sparkrun.core.limits import resolve_accelerator_memory_gb

    candidates = []
    hosts = placement.hosts_used if placement is not None else cluster.hosts
    for host in hosts:
        hw = cluster.hardware_for(host)
        if placement is not None:
            hw = hw.selected(placement.local_gpu_for_rank(r) for r in placement.ranks_on_host(host))
        for accel in hw.accelerators:
            capacity = resolve_accelerator_memory_gb(accel, hw)
            if capacity is None:
                return None, None
            candidates.append((capacity, accel.model))
    return min(candidates) if candidates else (None, None)


def display_vram_estimate(
    recipe,
    cli_overrides=None,
    auto_detect=True,
    cache_dir=None,
):
    """Display a standalone VRAM estimate for a recipe (no target hardware).

    For a placed launch, :func:`display_memory_plan` renders the estimate
    against the hosts it will run on.
    """
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
    if est.kv_arch_label:
        click.echo(f"  Architecture:     {est.kv_arch_label}")
    elif all([est.num_layers, est.num_kv_heads, est.head_dim]):
        click.echo(f"  Architecture:     {est.num_layers} layers, {est.num_kv_heads} KV heads, {est.head_dim} head_dim")
    click.echo(f"  Model weights:    {est.model_weights_gb:.2f} GiB")
    if est.model_max_len:
        click.echo(f"  Model context:    {est.model_max_len:,} tokens")
    if est.kv_cache_memory_bytes is not None:
        click.echo(f"  KV allocation:    {est.kv_cache_memory_bytes / 1024**3:.2f} GiB per GPU (explicit)")
    if est.kv_cache_total_gb is not None:
        # MLA's latent cache has no head dimension to shard, so it is duplicated
        # on every TP rank — worth saying, since raising --tp won't shrink it.
        replicated = " — replicated per rank" if est.kv_cache_replicated and est.tensor_parallel > 1 else ""
        click.echo(f"  KV demand estimate: {est.kv_cache_total_gb:.2f} GiB (max_model_len={est.max_model_len:,}){replicated}")
    click.echo(f"  Tensor parallel:  {est.tensor_parallel}")
    if est.pipeline_parallel > 1:
        click.echo(f"  Pipeline parallel: {est.pipeline_parallel}")
    click.echo(f"  Per-GPU total:    {est.total_per_gpu_gb:.2f} GiB")
    click.echo("  Memory fit:       UNKNOWN (target capacity unknown)")
    click.echo("  Allocation:       exclusive GPU by default; runtime verifies actual memory fit")
    if est.kv_cache_memory_bytes is not None:
        click.echo("  KV budget source: explicit bytes; gpu_memory_utilization does not size KV")
        if est.max_context_tokens is not None:
            click.echo(f"  Context estimate: {est.max_context_tokens:,} tokens within the explicit KV budget")
        else:
            click.echo("  Context capacity: unverified; runtime must size the cache")
    for w in est.warnings:
        click.echo(f"  Warning: {w}")


# ---------------------------------------------------------------------------
# Launch summary: hosts table, then the memory plan sized to those hosts
# ---------------------------------------------------------------------------

#: Provenance markers. No marker means "measured this run"; only exceptions
#: are marked, so a fully probed cluster reads clean.
_INVENTORY_MARK = "*"
_ESTIMATE_MARK = "~"
_DIFFERS_MARK = "!"


def _table(header: list[str], rows: list[list[str]], indent: str = "  ") -> list[str]:
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows, strict=True)]
    return [indent + "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip() for row in [header, *rows]]


def _mark(source: str | None) -> str:
    if source in (None, "detected"):
        return ""
    if source == "inventory":
        return _INVENTORY_MARK
    return _ESTIMATE_MARK


def _capacity_mark(verification: str) -> str:
    return {"detected": "", "inventory": _INVENTORY_MARK, "estimated": _ESTIMATE_MARK}.get(verification, "")


def _suffix(text: str, mark: str) -> str:
    return f"{text} {mark}" if mark else text


def format_host_table(
    assessment, host_list: list[str], placement=None, *, is_solo: bool = False, verbose: bool = False, dry_run: bool = False
) -> list[str]:
    """Render the launch hosts as one table plus a legend for any marked fact.

    *assessment* is :class:`~sparkrun.core.hardware_assessment.HardwareAssessment`.
    Columns that would say nothing are omitted (``RANKS`` unless a host runs
    several ranks). ``verbose`` appends the long-form evidence lines and the
    RDMA interface names.
    """
    facts = assessment.hosts
    multi_rank = placement is not None and any(len(placement.ranks_on_host(h)) > 1 for h in host_list)
    header = ["ROLE", "HOST"] + (["RANKS"] if multi_rank else []) + ["GPU", "MEMORY", "DRIVER", "RDMA"]

    def _driver(host: str) -> tuple[str | None, str]:
        drivers = facts.get(host, {}).get("drivers", {})
        if not drivers:
            return None, ""
        vendors = [a["vendor"] for a in facts[host]["accelerators"]]
        vendor = next((v for v in vendors if v in drivers), next(iter(drivers)))
        return drivers[vendor]["version"], drivers[vendor]["source"]

    head_driver = _driver(host_list[0])[0] if host_list else None
    used: set[str] = set()
    any_rdma = False
    rows = []
    for position, host in enumerate(host_list):
        fact = facts.get(host)
        role = "target" if is_solo else ("head" if position == 0 else "worker")
        row = [role, host]
        if multi_rank:
            row.append(str(len(placement.ranks_on_host(host))))
        if fact is None or not fact["accelerators"]:
            row += ["unknown", "unknown"]
        else:
            accels = fact["accelerators"]
            models = sorted({a["model"] for a in accels})
            gpu = "+".join(models) + (f" x{len(accels)}" if len(models) == 1 and len(accels) > 1 else "")
            gpu_mark = _mark(accels[0]["identity_source"])
            capacities = [a["capacity_gib"] for a in accels]
            if any(c is None for c in capacities):
                memory, memory_mark = "unknown", ""
            else:
                smallest = min(accels, key=lambda a: a["capacity_gib"])
                memory, memory_mark = f"{smallest['capacity_gib']:.1f} GiB", _capacity_mark(smallest["capacity_verification"])
            row += [_suffix(gpu, gpu_mark), _suffix(memory, memory_mark)]
            used.update(m for m in (gpu_mark, memory_mark) if m)
        version, source = _driver(host)
        if version is None:
            row.append("—")
        else:
            mark = _DIFFERS_MARK if head_driver is not None and version != head_driver else _mark(source)
            row.append(_suffix(version, mark))
            used.update([mark] if mark else [])
        interfaces = (fact or {}).get("interfaces", {})
        if interfaces.get("names"):
            any_rdma = True
            names = interfaces["names"]
            row.append(", ".join(names) if verbose else f"{len(names)} port{'s' if len(names) != 1 else ''}")
        else:
            row.append("none" if interfaces.get("status") == "not discovered" else "—")
        rows.append(row)

    lines = _table(header, rows)
    legend = []
    if _INVENTORY_MARK in used:
        legend.append(f"{_INVENTORY_MARK} from saved cluster inventory, not probed this run")
    if _ESTIMATE_MARK in used:
        if assessment.assumed:
            legend.append(f"{_ESTIMATE_MARK} assumed by platform policy, not probed" + (" (dry run doesn't probe)" if dry_run else ""))
        else:
            legend.append(f"{_ESTIMATE_MARK} platform default capacity, not measured")
    if _DIFFERS_MARK in used:
        legend.append(f"{_DIFFERS_MARK} differs from head")
    lines.extend("  " + line for line in legend)
    footer = []
    if not used and len(assessment.detected) == len(host_list):
        footer.append("All facts probed this run.")
    if any_rdma:
        footer.append("RDMA links not tested.")
    if footer:
        lines.append("  " + " ".join(footer))
    if verbose:
        for host, line in assessment.evidence.items():
            lines.append(f"  {host}: {line}")
    return lines


def _tokens(n: int) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1_000_000 else f"{n:,}"


def format_memory_plan(est, fit, *, hosts: int = 1) -> list[str]:
    """Render the per-GPU memory arithmetic, then the fit verdict, for a placed launch.

    *est* is the :class:`~sparkrun.models.vram.VRAMEstimate` sized to the
    smallest selected capacity; *fit* the
    :class:`~sparkrun.models.fit.FitResult` for the same placement. Each number
    appears once: per-host rows are added only when hosts differ or one fails.
    """
    shard = max(1, est.tensor_parallel * est.pipeline_parallel)
    shape = f"tp={est.tensor_parallel}" + (f", pp={est.pipeline_parallel}" if est.pipeline_parallel > 1 else "")
    sized = "sized to the smallest host above" if hosts > 1 else "sized to the host above"
    lines = [f"Memory per GPU ({shape}, {sized}):"]

    def row(label: str, value: str, detail: str = "") -> None:
        lines.append(f"  {label:<11}{value:>11}    {detail}".rstrip())

    dtype = est.model_dtype or "unknown dtype"
    if est.model_weights_gb > 0:
        row(
            "Weights",
            f"{est.model_weights_gb / shard:.1f} GiB",
            f"{est.model_weights_gb:.1f} GiB {dtype}" + (f" / {shard}" if shard > 1 else ""),
        )
    else:
        row("Weights", "unknown", "model size not detected")

    kv_kind = " ".join(p for p in (est.kv_dtype or "bf16", "MLA latent" if est.kv_arch == "mla" else None) if p)
    if est.kv_cache_memory_bytes is not None:
        row("KV cache", f"{est.kv_cache_memory_bytes / 1024**3:.1f} GiB", "explicit kv_cache_memory_bytes")
    elif est.kv_cache_per_gpu_gb is not None and est.max_model_len:
        replicated = ", replicated per rank" if est.kv_cache_replicated and est.tensor_parallel > 1 else ""
        row("KV demand", f"{est.kv_cache_per_gpu_gb:.1f} GiB", f"{kv_kind} at max_model_len {est.max_model_len:,}{replicated}")

    details = [d for d in fit.per_host.values() if d.accelerator_memory_gb is not None]
    limiting = min(details, key=lambda d: d.accelerator_memory_gb) if details else None
    if limiting is not None:
        cap = limiting.max_gpu_memory_utilization
        if limiting.nominal_memory_gb is not None and cap is not None and cap < 1.0:
            source = limiting.memory_limit_source or ""
            how = (
                "gpu_memory_utilization"
                if source == "runtime gpu_memory_utilization"
                else f"usable cap ({source})"
                if source
                else "usable cap"
            )
            budget_detail = f"{limiting.nominal_memory_gb:.1f} GiB x {cap:.0%} {how}"
        else:
            budget_detail = f"{limiting.accelerator_memory_gb:.1f} GiB, uncapped"
        row("Budget", f"{limiting.accelerator_memory_gb:.1f} GiB", budget_detail)

    if est.kv_cache_memory_bytes is not None:
        if est.max_context_tokens is not None:
            row("Context", f"{_tokens(est.max_context_tokens)}", "tokens within the explicit KV budget")
        else:
            row("Context", "unverified", "runtime sizes the cache")
    elif est.available_kv_gb is not None:
        detail = ""
        if est.available_kv_gb <= 0:
            detail = "none: the weights alone exceed the budget"
        elif est.max_context_tokens is not None:
            detail = f"~{_tokens(est.max_context_tokens)} tokens of {kv_kind} cache"
            if est.model_max_len and est.max_context_tokens > est.model_max_len:
                detail += f"; model limit {est.model_max_len:,}"
            elif est.context_multiplier is not None and est.max_model_len:
                detail += f" ({est.context_multiplier:.1f}x max_model_len)"
        row("KV space", f"{est.available_kv_gb:.1f} GiB", detail)
    elif limiting is not None and limiting.headroom_gb is not None:
        row("Spare", f"{limiting.headroom_gb:.1f} GiB", "after weights" + (" and KV demand" if est.kv_cache_per_gpu_gb else ""))

    if fit.status == "fits":
        row("Fit", "YES", "estimate; the runtime verifies actual memory")
    elif fit.status == "exceeds":
        row("Fit", "EXCEEDS", "on " + ", ".join(h for h, d in fit.per_host.items() if not d.ok))
    else:
        row("Fit", "UNVERIFIED", "; ".join(fit.unverified_reasons))
    if any(d.allocation_mode == "shared" for d in fit.per_host.values()):
        row("Allocation", "shared", "GPU shared with other workloads; the runtime verifies actual memory")

    distinct = {(round(d.accelerator_memory_gb or -1, 1), round(d.headroom_gb or 0, 1)) for d in fit.per_host.values()}
    if len(fit.per_host) > 1 and (len(distinct) > 1 or fit.status == "exceeds"):
        width = max(len(h) for h in fit.per_host)
        for host, d in fit.per_host.items():
            if d.accelerator_memory_gb is None:
                lines.append(f"    {host:<{width}}  capacity unknown")
            else:
                verdict = "" if d.ok else "  EXCEEDS"
                lines.append(f"    {host:<{width}}  {d.accelerator_memory_gb:>6.1f} GiB budget  {d.headroom_gb:>6.1f} GiB spare{verdict}")

    if est.context_multiplier is not None and est.context_multiplier < 1.0 and est.max_model_len:
        lines.append(f"  Warning: max_model_len {est.max_model_len:,} exceeds the KV space ({est.context_multiplier:.0%} fits)")
    for warning in est.warnings:
        lines.append(f"  Warning: {warning}")
    return lines


def display_memory_plan(recipe, *, cluster, placement=None, cli_overrides=None, auto_detect=True, cache_dir=None) -> None:
    """Estimate *recipe* against the smallest selected capacity and print :func:`format_memory_plan`."""
    from sparkrun.models.fit import check_fit

    target_mem, _target_model = _resolve_target_accelerator(cluster, placement)
    try:
        est = recipe.estimate_vram(
            cli_overrides=cli_overrides, auto_detect=auto_detect, cache_dir=cache_dir, total_gpu_memory_gb=target_mem
        )
    except Exception as e:
        click.echo(f"\nMemory estimation failed: {e}", err=True)
        return
    fit = check_fit(est, cluster, placement)
    hosts = len(placement.hosts_used) if placement is not None else len(cluster.hosts)
    click.echo()
    for line in format_memory_plan(est, fit, hosts=hosts):
        click.echo(line)


def format_monitor_table(
    data: dict[str, HostMonitorState],
    hosts: list[str],
) -> str:
    """Format cluster monitor data as a text table.

    Args:
        data: Mapping of host -> HostMonitorState with latest sample/error.
        hosts: Ordered list of hosts (determines row order).

    Returns:
        Formatted multi-line string.
    """
    # Widths are minimums; host column expands to fit longest hostname.
    host_w = max(16, max(map(len, hosts), default=0)) + 2

    header = f"{'HOST':<{host_w}}{'Jobs':>6}{'CPU%':>8}{'RAM%':>8}{'GPU%':>8}{'CPU Temp':>10}{'GPU Temp':>10}{'GPU Power':>11}"
    separator = "-" * len(header)

    lines = [header, separator]
    for host in hosts:
        state = data.get(host)
        if state is None or (state.latest is None and state.error is None):
            lines.append(f"{host:<{host_w}}{'(connecting...)':>6}")
            continue

        if state.latest is None:
            lines.append(f"{host:<{host_w}}{state.error or '(connecting...)'}")
            continue

        s = state.latest
        # Flag hosts with stale/reconnecting data.
        host_label = "%s (!)" % host if state.error else host
        jobs = s.sparkrun_jobs if s.sparkrun_jobs else "-"
        cpu_pct = s.cpu_usage_pct if s.cpu_usage_pct else "-"
        ram_pct = "%s%%" % s.mem_used_pct if s.mem_used_pct else "-"
        gpu_util = "%s" % s.gpu_util_pct if s.gpu_util_pct else "-"
        cpu_temp = "%s C" % s.cpu_temp_c if s.cpu_temp_c else "-"
        gpu_temp = "%s C" % s.gpu_temp_c if s.gpu_temp_c else "-"
        gpu_power = "%s W" % s.gpu_power_w if s.gpu_power_w else "-"

        lines.append(f"{host_label:<{host_w}}{jobs:>6}{cpu_pct:>8}{ram_pct:>8}{gpu_util:>8}{cpu_temp:>10}{gpu_temp:>10}{gpu_power:>11}")

    return "\n".join(lines)


def format_activity_table(frame, hosts: list[str]) -> str:
    """Format a :class:`~sparkrun.core.monitoring.MonitorFrame` as a text table.

    Like :func:`format_monitor_table` but sourced from a live-monitor frame:
    the ``Jobs`` column counts the workloads occupying the host from
    ``api.status`` (docker + local + provider), not the telemetry stream's
    docker-only count; telemetry columns come from ``activity.telemetry``.
    """
    host_w = max(16, max(map(len, hosts), default=0)) + 2

    header = f"{'HOST':<{host_w}}{'Jobs':>6}{'CPU%':>8}{'RAM%':>8}{'GPU%':>8}{'CPU Temp':>10}{'GPU Temp':>10}{'GPU Power':>11}"
    separator = "-" * len(header)

    lines = [header, separator]
    for host in hosts:
        activity = frame.for_host(host) if frame is not None else None
        if activity is None:
            lines.append(f"{host:<{host_w}}{'(connecting...)':>6}")
            continue

        # Jobs from occupancy (all executors), not the docker-only telemetry count.
        n_jobs = sum(len(w.containers) or 1 for w in activity.workloads)
        jobs = str(n_jobs) if activity.workloads else "-"

        s = activity.telemetry
        if s is None:
            err = activity.telemetry_error or activity.status_error
            host_label = "%s (!)" % host if err else host
            lines.append(f"{host_label:<{host_w}}{jobs:>6}{'-':>8}{'-':>8}{'-':>8}{'-':>10}{'-':>10}{'-':>11}")
            continue

        host_label = "%s (!)" % host if (activity.telemetry_error or activity.status_error) else host
        cpu_pct = s.cpu_usage_pct if s.cpu_usage_pct else "-"
        ram_pct = "%s%%" % s.mem_used_pct if s.mem_used_pct else "-"
        gpu_util = "%s" % s.gpu_util_pct if s.gpu_util_pct else "-"
        cpu_temp = "%s C" % s.cpu_temp_c if s.cpu_temp_c else "-"
        gpu_temp = "%s C" % s.gpu_temp_c if s.gpu_temp_c else "-"
        gpu_power = "%s W" % s.gpu_power_w if s.gpu_power_w else "-"

        lines.append(f"{host_label:<{host_w}}{jobs:>6}{cpu_pct:>8}{ram_pct:>8}{gpu_util:>8}{cpu_temp:>10}{gpu_temp:>10}{gpu_power:>11}")

    return "\n".join(lines)


def format_validation_report(qualified_name: str, issues: list, *, title: str | None = None) -> str:
    """Render recipe-validation findings for a terminal.

    Shared by ``recipe validate`` and the launch paths, which differ only in
    *title*: under ``recipe validate`` the context is obvious, whereas amid a
    launch the reader needs telling that these came from validating the recipe
    rather than from anything that has started.

    Validation messages are paragraphs, not labels — each one names the
    defect, the consequence *and* the fix, so run together at full width they
    read as a wall.  Three things break that up:

    * a scannable header line per finding (severity + check name), and a blank
      line between findings, so the number of distinct problems is visible at a
      glance rather than parsed out of the prose;
    * the diagnosis wrapped and indented beneath it;
    * :attr:`~sparkrun.core.validation.RecipeIssue.fix` as its own further-
      indented block, because it is read at a different moment than the
      diagnosis — only once you've decided you care.

    Wrapping is capped at 100 columns even on a wide terminal: a 200-column
    paragraph is no easier to read than an unwrapped one.  Colour goes through
    ``click.style``, which is a no-op when stdout isn't a TTY, so piping to a
    file or a grep yields plain text.
    """
    import shutil

    width = max(40, min(shutil.get_terminal_size(fallback=(80, 24)).columns, 100))
    indent = "    "

    from sparkrun.core.validation import ERROR, SEVERITIES, SUGGESTION, WARNING

    # Label, colour and bold per severity.  ERROR shouts (uppercase + bold)
    # because it is the only one that always stops something; suggestion is
    # dimmed so a long report reads as a hierarchy rather than a list.
    styling = {
        ERROR: ("ERROR", "red", True, False),
        WARNING: ("warning", "yellow", False, False),
        SUGGESTION: ("suggestion", "cyan", False, True),
    }
    counts = []
    for level in SEVERITIES:
        n = sum(1 for i in issues if i.severity == level)
        if n:
            counts.append("%d %s%s" % (n, level, "" if n == 1 else "s"))

    lines = ["%s: %s" % (title or "Recipe '%s'" % qualified_name, ", ".join(counts))]
    for issue in issues:
        lines.append("")
        label, colour, bold, dim = styling.get(issue.severity, (issue.severity, None, False, False))
        lines.append(
            "%s  %s"
            % (
                click.style(label, fg=colour, bold=bold, dim=dim),
                click.style(issue.code, bold=not dim, dim=dim),
            )
        )
        lines.extend(_wrap(issue.summary, width, indent))
        if issue.fix:
            lines.append("")
            lines.extend(_wrap(issue.fix, width, indent + "  "))
    return "\n".join(lines)


def _wrap(text: str, width: int, indent: str) -> list[str]:
    import textwrap

    return textwrap.wrap(
        text,
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def timing_tree_depth_for_verbosity(verbosity: int | bool = 0) -> int | None:
    """Map global CLI verbosity to the standard timing-tree depth.

    Default output shows two levels, ``-v`` shows three, ``-vv`` shows four,
    and ``-vvv`` (or greater) removes the limit. Quiet mode uses the compact
    default depth if a caller explicitly requests a timing table.
    """
    if isinstance(verbosity, bool):
        verbosity = 1 if verbosity else 0
    return None if verbosity >= 3 else max(2, 2 + verbosity)


#: Rank-local Docker-start metrics: key → observation timestamp field, in
#: report order.  ``key`` matches the ``serve.startup_<key>`` span names, so a
#: reader can move between the block, the tree and the exported timeline
#: without a translation table.
_STARTUP_METRICS = (
    ("port_open", "port_open_unix_ns"),
    ("http_ready", "http_ready_unix_ns"),
    ("ttft", "first_token_unix_ns"),
)

#: Row labels for :func:`format_startup_readiness`.  The live readiness line
#: words them differently (it carries one shared "container-start" prefix), so
#: only the *values* are shared between the two renderings.
_STARTUP_LABELS = {"port_open": "TTR port-open", "http_ready": "TTR HTTP-ready", "ttft": "TTFT"}

#: Timeline spans whose numbers :func:`format_startup_readiness` renders.  A
#: caller printing that block passes these as ``format_launch_timings(omit=…)``
#: so one launch does not report the same three figures twice, once in a table
#: built for them and once as rows that are not terms in its total.
STARTUP_SPAN_NAMES = frozenset("serve.startup_%s" % key for key, _ in _STARTUP_METRICS)


def startup_readiness_durations(observation: dict) -> dict[str, str]:
    """Render a rank-local startup observation's three elapsed times.

    Returns ``{"port_open": "6.081s", "http_ready": "15.682s", "ttft": …}``.
    Every value is elapsed from the serving container's start, so the three
    **overlap and must never be summed**.

    Shared by the live readiness line and the end-of-launch block precisely
    because the user is invited to compare them: two renderings computing the
    same figure independently is how they come to disagree.

    A missing timestamp renders ``"unavailable"`` rather than ``0`` — the
    observation omits what it did not observe — and an endpoint-only
    observation (``readiness.inference: false``) renders TTFT as not
    applicable rather than as a failure to measure it.
    """
    start = observation["container_started_unix_ns"]
    endpoint_only = observation.get("inference_requested") is False
    rendered: dict[str, str] = {}
    for key, field_name in _STARTUP_METRICS:
        if key == "ttft" and endpoint_only:
            rendered[key] = "not applicable (inference disabled)"
        elif field_name in observation:
            rendered[key] = "%.3fs" % ((observation[field_name] - start) / 1e9)
        else:
            rendered[key] = "unavailable"
    return rendered


def format_startup_readiness(
    observation: dict | None,
    *,
    host: str | None = None,
    width: int = 62,
    title: str = "Startup readiness",
) -> str:
    """Render the rank-local Docker-start metrics as an end-of-launch block.

    The same three numbers the readiness line reports the instant the endpoint
    answers.  That line is emitted while ``docker logs -f`` is still writing,
    so by the time a launch finishes it is thousands of lines up the
    scrollback — which is the whole reason it is repeated beside the timing
    tree once the stream has stopped.

    A **block** rather than three rows in the tree, because these are
    cumulative from one origin and overlap: the tree's siblings are additive,
    and rows there that must not be summed would read as a stage breakdown.
    They do also reach the tree, as foreign-clock spans marked non-additive —
    that is for the consumer reading the exported timeline, where a
    machine-readable marker is the only thing that can carry the warning.

    Returns ``""`` when there is no observation.  A legacy endpoint wait
    measures nothing from container start, and the two stages it *does*
    measure are already ``serve.port_open`` / ``serve.health_ok`` in the tree.
    """
    if not isinstance(observation, dict) or type(observation.get("container_started_unix_ns")) is not int:
        return ""
    where = "rank 0" + (" on %s" % host if host else "")
    lines = ["%s (%s, elapsed from container start):" % (title, where)]
    durations = startup_readiness_durations(observation)
    for key, _ in _STARTUP_METRICS:
        label = _STARTUP_LABELS[key]
        pad = max(1, width - 2 - len(label))
        lines.append("  %s%s%s" % (label, " " * pad, durations[key]))
    measurement = observation.get("measurement")
    if measurement:
        lines.append("  measurement: %s" % measurement)
    return "\n".join(lines)


def format_launch_timings(
    export: dict,
    *,
    width: int = 62,
    title: str = "Launch timings",
    max_depth: int | None = None,
    omit: "Container[str]" = frozenset(),
) -> str:
    """Render a :meth:`~sparkrun.core.timing.Timeline.export` as a tree.

    Spans nest by parent id rather than by name, because fan-out spans
    repeat their name (one per host / per distribution entry) and a
    name-keyed tree would collapse them into one row.

    ``max_depth`` counts displayed levels starting at one for root spans.
    Deeper descendants are omitted. ``None`` leaves the tree unbounded.

    ``omit`` drops spans by name, **with their descendants** — for a caller
    that has already presented them better elsewhere
    (:data:`STARTUP_SPAN_NAMES` beside :func:`format_startup_readiness` is
    the case this exists for).  Display only: the spans stay in the export,
    which is what the diagnostics record and benchmark metadata read.

    Worth being deliberate about, because a row here reads as a *term*: the
    tree's rows sum to its total, so leaving a non-additive span in it either
    misleads or needs an apology stapled to the row.  Dropping one whose
    figures appear nowhere else would be the opposite mistake, which is why
    this is the caller's call and not a rule keyed off ``composition``.
    """
    if max_depth is not None and max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    spans = export.get("spans") or []
    if not spans:
        return ""

    children: dict[int | None, list[dict]] = {}
    for span in spans:
        if span.get("name") in omit:
            continue
        children.setdefault(span.get("parent"), []).append(span)
    for group in children.values():
        group.sort(key=lambda s: (s.get("t_start", 0.0), s.get("id", 0)))

    lines = ["%s (total %.1fs):" % (title, export.get("duration_s", 0.0))]

    def walk(parent: int | None, depth: int) -> None:
        for span in children.get(parent, ()):
            attrs = span.get("attrs") or {}
            label = attrs.get("label") or span.get("name", "")
            indent = "  " * (depth + 1)
            status = span.get("status", "ok")
            if status == "skipped":
                reason = attrs.get("reason")
                timing = "skipped" + (" (%s)" % reason if reason else "")
            elif status == "open":
                # Never closed: the launch raised while it was running, so
                # this is where it stopped.  Reporting the elapsed time
                # alone would read as a completed stage.
                timing = "%.1fs (did not finish)" % span.get("duration_s", 0.0)
            else:
                timing = "%.1fs" % span.get("duration_s", 0.0)
                if status == "error":
                    timing += " (failed)"
            # A foreign-clock span was measured on another machine, so its
            # duration is not additive with its neighbours' and its position
            # carries the skew.  Naming the clock is what stops the tree
            # reading as one arithmetic whole.
            clock = span.get("clock")
            if clock:
                timing += "  [%s]" % clock
            if attrs.get("composition") == "non_additive":
                semantics = str(attrs.get("timing_semantics") or "aggregate").replace("_", " ")
                timing += "  [%s; non-additive]" % semantics
            pad = max(1, width - len(indent) - len(label))
            lines.append("%s%s%s%s" % (indent, label, " " * pad, timing))
            displayed_depth = depth + 1
            if max_depth is None or displayed_depth < max_depth:
                walk(span.get("id"), depth + 1)

    walk(None, 0)
    return "\n".join(lines)
