"""``sparkrun.api.benchmark`` — public Python entry point for benchmark runs.

This module is currently a *thin wrapper* over the CLI orchestration in
``sparkrun.cli._benchmark._run_benchmark``.  It establishes the API contract
(``BenchmarkOptions`` in, ``BenchmarkResult`` out, typed exceptions on
failure) so callers and the future arena/CLI delegation can target it now.

Step 7 of the redesign moves the orchestration body into this module; the
wrapper shape remains stable across that refactor.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sparkrun.api._benchmark_models import (
    BenchmarkOptions,
    BenchmarkResult,
    ResumeMode,
)
from sparkrun.api._context import resolve_sctx
from sparkrun.api._errors import BenchmarkFailed, SparkrunError

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext

logger = logging.getLogger(__name__)


def benchmark(
    options: BenchmarkOptions,
    *,
    sctx: "SparkrunContext | None" = None,
) -> BenchmarkResult:
    """Run a benchmark and return a structured :class:`BenchmarkResult`.

    Args:
        options: Inputs for the benchmark run.
        sctx: Optional shared :class:`SparkrunContext`.  When omitted a
            fresh session is built; callers chaining multiple ``api.*``
            calls can construct one ``sctx`` and pass it to share state.

    Raises:
        :class:`BenchmarkFailed`: The run terminated unsuccessfully
            (non-zero exit, task failures, or aborted launch).
        :class:`SparkrunError` (subclass): Other typed failures.
        :class:`KeyboardInterrupt`: Re-raised after the underlying flow
            persists its state.
    """
    import click

    sctx = resolve_sctx(sctx)

    # Build a minimal Click context so the existing CLI orchestration can
    # call ``_get_context(ctx)`` unchanged.  We construct an ephemeral
    # ``click.Command`` and inject our sctx via ``ctx.obj``.
    fake_cmd = click.Command(name="api.benchmark")
    ctx = click.Context(fake_cmd, obj={"sparkrun_ctx": sctx})

    from sparkrun.cli._benchmark import _run_benchmark

    if options.progress_callback is not None:
        logger.debug(
            "BenchmarkOptions.progress_callback is set but step 3 does not wire it. Callback will be honoured after step 7 lift-and-shift."
        )

    fresh, submission_id_for_extras = _translate_lifecycle(options)
    kwargs = _build_run_benchmark_kwargs(options, fresh, submission_id_for_extras)

    try:
        with ctx:
            bench_result = _run_benchmark(ctx, **kwargs)
    except KeyboardInterrupt:
        raise
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        raise BenchmarkFailed("benchmark run aborted (exit %d)" % code, exit_code=code) from exc
    except SparkrunError:
        raise
    except Exception as exc:
        raise SparkrunError("benchmark failed: %s" % exc) from exc

    return _build_result(options, bench_result)


def _translate_lifecycle(options: BenchmarkOptions) -> tuple[bool, str | None]:
    """Translate API-shaped lifecycle settings.

    Returns ``(fresh_legacy, submission_id)``. ``fresh_legacy`` is kept for
    back-compat with callers reading the old kwarg; ``resume_mode`` is now
    threaded through directly.
    """
    fresh_legacy = options.resume == ResumeMode.FRESH
    submission_id = options.state_extras.get("submission_id") if options.state_extras else None
    return fresh_legacy, submission_id


def _build_run_benchmark_kwargs(
    options: BenchmarkOptions,
    fresh: bool,
    submission_id_for_extras: str | None,
) -> dict[str, Any]:
    """Translate ``BenchmarkOptions`` to keyword args for ``_run_benchmark``.

    Unmapped CLI-only kwargs use the same defaults the click decorators
    would have supplied.
    """
    recipe_name: str
    if isinstance(options.recipe, str):
        recipe_name = options.recipe
    else:
        recipe_name = getattr(options.recipe, "qualified_name", None) or str(options.recipe)

    cluster_name: str | None = None
    if isinstance(options.cluster, str):
        cluster_name = options.cluster
    elif options.cluster is not None:
        cluster_name = getattr(options.cluster, "name", None)

    image = options.overrides.get("image") if isinstance(options.overrides, dict) else None
    port = options.overrides.get("port") if isinstance(options.overrides, dict) else None

    bench_options_tuple = tuple("%s=%s" % (k, v) for k, v in options.bench_args.items())
    executor_args_tuple = tuple(options.extra_docker_opts) if options.extra_docker_opts else ()

    # Arena defaults: when options.arena is True, supply the pinned profile and
    # performance category when the caller has not specified them explicitly.
    # Auth and upload are CLI-only concerns; the API caller is responsible for
    # those when using options.arena=True.
    effective_profile = options.profile
    effective_category = options.category
    if options.arena:
        if not effective_profile:
            from sparkrun.cli._arena_flow import ARENA_BENCHMARK_PROFILE

            effective_profile = ARENA_BENCHMARK_PROFILE
        if not effective_category:
            effective_category = "performance"

    return {
        "recipe_name": recipe_name,
        "hosts": list(options.hosts) if options.hosts else [],
        "hosts_file": None,
        "cluster_name": cluster_name,
        "tensor_parallel": None,
        "pipeline_parallel": None,
        "data_parallel": None,
        "gpu_mem": None,
        "max_model_len": None,
        "options": (),
        "image": image,
        "solo": options.solo,
        "port": port,
        "profile": effective_profile,
        "framework": options.framework,
        "output_file": options.output_file,
        "bench_options": bench_options_tuple,
        "api_key_env": options.api_key_env,
        "exit_on_first_fail": options.exit_on_first_fail,
        "no_stop": options.no_stop,
        "skip_run": options.skip_run,
        "sync_tuning": options.sync_tuning,
        "rootful": options.rootful,
        "bench_timeout": options.timeout,
        "dry_run": options.dry_run,
        "executor_args": executor_args_tuple,
        "extra_args": (),
        "export_results_files": options.export_files,
        "fresh": fresh,
        "resume_mode": options.resume,
        "on_prompt_required": options.on_prompt_required,
        "submission_id_for_extras": submission_id_for_extras,
        "scheduler_name": options.scheduler,
        "category": effective_category,
    }


def _build_result(options: BenchmarkOptions, bench_result: Any) -> BenchmarkResult:
    """Translate the CLI's internal ``BenchmarkResult`` into the API one.

    Best-effort population: fields the internal result doesn't yet carry
    surface as sensible defaults; step 7 will plumb the rest.

    Note: the internal ``BenchmarkResult.framework`` field is a
    ``BenchmarkingPlugin`` object (not a string); we extract
    ``.framework_name`` from it where present.
    """
    # --- outputs ---
    outputs: dict[str, str] = {}
    raw_outputs = getattr(bench_result, "outputs", None) or {}
    for k, v in raw_outputs.items():
        if v is not None:
            outputs[k] = str(v)

    # --- framework (plugin object → name string) ---
    framework_plugin = getattr(bench_result, "framework", None)
    if framework_plugin is not None and not isinstance(framework_plugin, str):
        framework_str = getattr(framework_plugin, "framework_name", None) or str(framework_plugin)
    else:
        framework_str = framework_plugin or options.framework or ""

    # --- category: prefer explicit option, then plugin's primary_category ---
    category = options.category or ""
    if not category and framework_plugin is not None and not isinstance(framework_plugin, str):
        category = getattr(framework_plugin, "primary_category", "") or ""
    if not category and framework_str:
        # Fallback: try to load the framework plugin to get its primary_category.
        try:
            from sparkrun.core.bootstrap import get_benchmarking_framework

            fw = get_benchmarking_framework(framework_str)
            category = getattr(fw, "primary_category", "") or ""
        except Exception:
            pass

    # --- container image details ---
    container_image_raw = getattr(bench_result, "container_image", None)
    container_image_str = str(container_image_raw) if container_image_raw else ""

    # The internal BenchmarkResult does not currently carry sha / pinned /
    # longterm fields — getattr defaults handle gracefully.
    return BenchmarkResult(
        success=bool(getattr(bench_result, "success", False)),
        benchmark_id=str(getattr(bench_result, "benchmark_id", "") or ""),
        category=category,
        framework=framework_str,
        profile=getattr(bench_result, "profile", None) or options.profile,
        results=dict(getattr(bench_result, "results", None) or {}),
        outputs=outputs,
        run_result=None,  # step 7
        cluster_id=str(getattr(bench_result, "cluster_id", "") or ""),
        host_list=tuple(getattr(bench_result, "host_list", ()) or ()),
        container_image=container_image_str,
        container_image_sha=getattr(bench_result, "container_image_sha", None),
        container_image_sha_pinned=bool(getattr(bench_result, "container_image_sha_pinned", False)),
        container_image_longterm_ref=getattr(bench_result, "longterm_image_ref", None),
        container_image_longterm_pinned=bool(getattr(bench_result, "longterm_image_pinned", False)),
        metadata={
            "framework": framework_str,
            "profile": getattr(bench_result, "profile", None) or options.profile,
            "bench_args": dict(getattr(bench_result, "benchmark_args", None) or options.bench_args),
        },
        state_dir=getattr(bench_result, "state_dir", None),
        resumed=bool(getattr(bench_result, "resumed", False)),
        submission_id=getattr(bench_result, "submission_id", None),
    )


__all__ = ["benchmark"]
