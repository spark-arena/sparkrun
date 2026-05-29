"""sparkrun benchmark command — run benchmarks against inference recipes.

CLI presentation shell.  Orchestration lives in ``sparkrun.api._benchmark``.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable, TYPE_CHECKING

import click

from ._common import (
    PROFILE_NAME,
    RECIPE_NAME,
    _display_vram_estimate,
    _get_context,
    dry_run_option,
    host_options,
    recipe_override_options,
    with_host_context,
    HIDE_ADVANCED_OPTIONS,
)

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_TIMEOUT: int = 14400  # 4 hours

if TYPE_CHECKING:
    from sparkrun.api._benchmark_models import ResumeMode


def _resolve_resume_prompt(
    state,
    total_tasks: int,
    on_prompt_required: "Callable[[Any], bool] | None",
) -> bool:
    """Decide whether to resume incomplete state when mode is AUTO.

    Resolution order:
    1. Explicit ``on_prompt_required`` callback — caller drives the answer.
    2. TTY: ``click.confirm`` with default=True (preserves existing CLI UX).
    3. Non-TTY: default to True (resume) — sensible non-interactive default.
    """
    if on_prompt_required is not None:
        return bool(on_prompt_required(state))
    import sys as _sys

    if _sys.stdin.isatty():
        return click.confirm(
            "Found existing incomplete benchmark state (%d/%d tasks done). Resume?" % (len(state.completed_indices), total_tasks),
            default=True,
        )
    return True


class _BenchmarkGroup(click.Group):
    """Click group for ``sparkrun benchmark`` with:

    - Backwards-compat fallback so ``sparkrun benchmark <recipe>`` keeps
      working (routes to ``performance`` if registered, else ``run``).
    - Name aliases: ``perf`` → ``performance``.
    """

    _ALIASES: dict[str, str] = {"perf": "performance"}

    def get_command(self, ctx, cmd_name):
        cmd_name = self._ALIASES.get(cmd_name, cmd_name)
        # Lazy: re-register category commands if they aren't present yet.
        if cmd_name not in self.commands and cmd_name in self._categories_dynamic():
            _register_category_commands(self)
        return super().get_command(ctx, cmd_name)

    def resolve_command(self, ctx, args):
        if args and args[0] in self._ALIASES:
            args = list(args)
            args[0] = self._ALIASES[args[0]]
        return super().resolve_command(ctx, args)

    def _categories_dynamic(self) -> set[str]:
        try:
            from sparkrun.core.bootstrap import list_benchmark_categories

            return set(list_benchmark_categories())
        except Exception:
            return set()

    def parse_args(self, ctx, args):
        # Ensure category subcommands are registered before we use ``self.commands``
        # to decide what counts as a "known" subcommand vs. a fallback recipe name.
        # Lazy registration here keeps SAF bootstrapping out of module-import time
        # (which would pollute the autouse ``isolate_stateful`` test fixture).
        _register_category_commands(self)
        # If none of the known subcommand names (or known aliases) appears as a
        # standalone token in the argument list, prepend the fallback target so
        # legacy ``sparkrun benchmark <recipe>`` keeps working.
        # Updated target: ``performance`` when registered, else ``run``.
        if args:
            all_known = set(self.commands) | set(self._ALIASES)
            has_subcommand = any(a in all_known for a in args if not a.startswith("-"))
            if not has_subcommand:
                fallback = "performance" if "performance" in self.commands else "run"
                args = [fallback] + list(args)
        return super().parse_args(ctx, args)


@click.group(cls=_BenchmarkGroup)
@click.pass_context
def benchmark(ctx):
    """Benchmark an inference recipe.

    Runs the full benchmark flow: launch inference, run benchmark, stop
    inference.

    Manage benchmark profiles via the registry subcommands:

      sparkrun registry list-benchmark-profiles

      sparkrun registry show-benchmark-profile <name>

    Examples:

      sparkrun benchmark qwen3-1.7b-sglang --solo

      sparkrun benchmark qwen3-1.7b-sglang --tp 2 --profile spark-arena-v1

      sparkrun benchmark qwen3-1.7b-sglang -b depth=0,2048,4096 -b tg=32,128

      sparkrun benchmark qwen3-1.7b-sglang --skip-run --solo
    """
    pass


def _shared_run_options(f):
    """All Click options/arguments shared by ``benchmark run`` and the per-category
    subcommands.  Apply as a single decorator so the option stack stays in one place.
    """
    decorators = [
        click.argument("recipe_name", type=RECIPE_NAME),
        host_options,
        recipe_override_options,
        click.option("--solo", is_flag=True, help="Force single-node mode"),
        click.option("--port", type=int, default=None, help="Override serve port"),
        click.option("--profile", default=None, type=PROFILE_NAME, help="Benchmark profile name or file path"),
        click.option("--framework", default=None, help="benchmarking framework (default from config.default_benchmark_framework)"),
        click.option("--output", "output_file", default=None, type=click.Path(), help="Output file for results YAML"),
        click.option("-b", "--benchmark-option", "bench_options", multiple=True, help="Override benchmark arg: -b key=value (repeatable)"),
        click.option(
            "--api-key-env",
            default=None,
            help="Name of the environment variable to read the API key from (e.g. OPENAI_API_KEY).",
        ),
        click.option(
            "--exit-on-first-fail/--no-exit-on-first-fail",
            "exit_on_first_fail",
            default=True,
            help="Abort benchmark on first failure and skip saving results (default: enabled)",
        ),
        click.option("--no-stop", is_flag=True, help="Don't stop inference after benchmarking"),
        click.option("--skip-run", is_flag=True, help="Skip launching inference (benchmark existing instance)"),
        click.option("--sync-tuning", is_flag=True, help="Sync tuning configs from registries before benchmarking"),
        click.option("--rootful", is_flag=True, help="Run with --privileged as root inside container (legacy behavior)"),
        click.option(
            "--timeout",
            "bench_timeout",
            type=int,
            default=None,
            help="Benchmark timeout in seconds (default: %d, or from profile)" % DEFAULT_BENCHMARK_TIMEOUT,
        ),
        click.option("--fresh", is_flag=True, default=False, help="Force fresh start, deleting prior state if any"),
        click.option(
            "--resume",
            "resume_flag",
            is_flag=True,
            default=False,
            help="Non-interactive resume: if existing state matches, resume from there; otherwise start fresh.",
        ),
        click.option(
            "--arena",
            "arena_flag",
            is_flag=True,
            default=False,
            help="Submit results to Spark Arena (requires `sparkrun arena login`).",
        ),
        click.option(
            "--local-test",
            "local_test",
            is_flag=True,
            default=False,
            hidden=HIDE_ADVANCED_OPTIONS,
            help="Arena local-test mode: skip auth and upload, simulate end-to-end.",
        ),
        dry_run_option,
        click.option(
            "--scheduler",
            "scheduler_name",
            default=None,
            help="Registered scheduler name (e.g. 'greedy', 'occupancy-sparse', 'occupancy-dense'). Defaults to the recipe's scheduler field, then 'greedy'.",
            hidden=HIDE_ADVANCED_OPTIONS,
        ),
        click.option(
            "--executor-args",
            multiple=True,
            hidden=HIDE_ADVANCED_OPTIONS,
            help="Arguments passed directly to the container executor (e.g. docker run)",
        ),
        click.argument("extra_args", nargs=-1, type=click.UNPROCESSED),
    ]
    # Apply in reverse so the first item in the list ends up outermost (Click
    # applies decorators bottom-up; reversing preserves declaration order).
    for dec in reversed(decorators):
        f = dec(f)
    return f


def _invoke_benchmark(ctx, *, category, **kwargs):
    """Shared body for ``benchmark run`` / per-category subcommands.

    Resolves the ResumeMode from the ``resume_flag``/``fresh`` kwargs, then
    delegates to ``_run_benchmark`` with the pinned *category*.

    When ``arena_flag`` is True, calls preflight_arena before the benchmark
    and finalize_arena after a successful run.
    """
    from sparkrun.api._benchmark_models import ResumeMode

    resume_flag = kwargs.pop("resume_flag", False)
    fresh = kwargs.get("fresh", False)
    arena_flag = kwargs.pop("arena_flag", False)
    local_test = kwargs.pop("local_test", False)

    if resume_flag and fresh:
        raise click.UsageError("--resume and --fresh are mutually exclusive")
    if resume_flag:
        _resume_mode = ResumeMode.IF_EXISTS
    elif fresh:
        _resume_mode = ResumeMode.FRESH
    else:
        _resume_mode = ResumeMode.AUTO

    # Arena preflight: when --arena is set, do auth and submission_id generation
    # before the benchmark runs so the same id flows through state.extras.
    arena_submission_id: str | None = None
    if arena_flag:
        from sparkrun.cli._arena_flow import preflight_arena

        arena_submission_id, arena_profile = preflight_arena(local_test=local_test, ctx=ctx)
        # Only override profile when user did not supply one explicitly
        if not kwargs.get("profile") and arena_profile:
            kwargs["profile"] = arena_profile

    dry_run = kwargs.get("dry_run", False)

    bench_result = _run_benchmark(
        ctx,
        recipe_name=kwargs.pop("recipe_name"),
        hosts=kwargs.pop("hosts"),
        hosts_file=kwargs.pop("hosts_file"),
        cluster_name=kwargs.pop("cluster_name"),
        tensor_parallel=kwargs.pop("tensor_parallel"),
        pipeline_parallel=kwargs.pop("pipeline_parallel"),
        data_parallel=kwargs.pop("data_parallel"),
        gpu_mem=kwargs.pop("gpu_mem"),
        max_model_len=kwargs.pop("max_model_len"),
        options=kwargs.pop("options"),
        image=kwargs.pop("image"),
        solo=kwargs.pop("solo"),
        port=kwargs.pop("port"),
        profile=kwargs.pop("profile"),
        framework=kwargs.pop("framework"),
        output_file=kwargs.pop("output_file"),
        bench_options=kwargs.pop("bench_options"),
        api_key_env=kwargs.pop("api_key_env"),
        exit_on_first_fail=kwargs.pop("exit_on_first_fail"),
        no_stop=kwargs.pop("no_stop"),
        skip_run=kwargs.pop("skip_run"),
        sync_tuning=kwargs.pop("sync_tuning"),
        rootful=kwargs.pop("rootful"),
        bench_timeout=kwargs.pop("bench_timeout"),
        dry_run=kwargs.pop("dry_run"),
        executor_args=kwargs.pop("executor_args"),
        extra_args=kwargs.pop("extra_args"),
        fresh=fresh,
        resume_mode=_resume_mode,
        scheduler_name=kwargs.pop("scheduler_name"),
        host_list=kwargs.pop("host_list", None),
        cluster_mgr=kwargs.pop("cluster_mgr", None),
        category=category,
        submission_id_for_extras=arena_submission_id,
    )

    # Arena finalize: persist extras and upload (unless dry_run/local_test).
    if arena_flag and bench_result and getattr(bench_result, "success", False):
        from sparkrun.cli._arena_flow import finalize_arena

        finalize_arena(
            ctx=ctx,
            bench_result=bench_result,
            submission_id=arena_submission_id,
            local_test=local_test,
            dry_run=dry_run,
        )

    return bench_result


@benchmark.command("run")
@_shared_run_options
@click.pass_context
@with_host_context
def benchmark_run(ctx, **kwargs):
    """Run a benchmark against an inference recipe."""
    return _invoke_benchmark(ctx, category=None, **kwargs)


def _make_category_command(category: str, *, doc: str | None = None):
    """Build a Click command bound to a specific benchmark category.

    The command shares the entire ``_shared_run_options`` stack with
    ``benchmark run`` and pins the *category* so framework defaulting /
    validation runs in ``_run_benchmark``.
    """

    @click.command(category)
    @_shared_run_options
    @click.pass_context
    @with_host_context
    def _cmd(ctx, **kwargs):
        return _invoke_benchmark(ctx, category=category, **kwargs)

    _cmd.name = category
    _cmd.__doc__ = doc or "Run a %s benchmark." % category
    return _cmd


def _register_category_commands(group):
    """Register a subcommand for every registered benchmark category.

    Categories with no plugins are skipped; registration is idempotent.
    """
    try:
        from sparkrun.core.bootstrap import list_benchmark_categories

        cats = list_benchmark_categories()
    except Exception:
        return
    for cat in cats:
        if cat in group.commands:
            continue
        group.add_command(_make_category_command(cat), name=cat)


# Per-category subcommand registration happens lazily on the first
# ``benchmark`` invocation (see ``_BenchmarkGroup.parse_args`` and
# ``get_command``) so importing this module never bootstraps SAF — that
# would pollute test fixtures and double-init the plugin registry.


@benchmark.command("resume")
@click.argument("benchmark_id")
@dry_run_option
@click.pass_context
def benchmark_resume(ctx, benchmark_id, dry_run):
    """Resume a paused benchmark by id."""
    sctx = _get_context(ctx)
    _resume_benchmark_run(ctx, benchmark_id, dry_run, sctx=sctx)


def _resume_benchmark_run(ctx, benchmark_id: str, dry_run: bool, *, sctx=None):
    """Thin CLI shell over ``sparkrun.api._benchmark.resume_benchmark``.

    Shared by ``benchmark resume`` and ``arena benchmark resume``.  Sets up
    the CLI progress emitter, delegates the orchestration to the API, and
    translates typed exceptions into ``click.echo`` + ``sys.exit``.  Returns
    the ``results`` mapping (keys: ``rows``, ``csv``, ``json``, etc.) on
    success; otherwise ``sys.exit`` is called and this never returns.
    """
    from sparkrun.api._benchmark import resume_benchmark, _ProgressEmitter
    from sparkrun.api._errors import BenchmarkFailed, NoResumableState, SparkrunError

    if sctx is None:
        sctx = _get_context(ctx)

    class _CliEmitter(_ProgressEmitter):
        def banner(self, line: str) -> None:
            click.echo(line)

        def info(self, msg: str) -> None:
            click.echo(msg)

        def warning(self, msg: str) -> None:
            click.echo("Warning: %s" % msg, err=True)

        def error(self, msg: str) -> None:
            click.echo("Error: %s" % msg, err=True)

    try:
        return resume_benchmark(benchmark_id, dry_run=dry_run, sctx=sctx, emitter=_CliEmitter())
    except KeyboardInterrupt:
        sys.exit(130)
    except BenchmarkFailed as e:
        # exit_code 0 is the "already complete" case — print the message on
        # stdout (not stderr) to preserve the prior CLI behaviour.
        if e.exit_code == 0:
            click.echo(str(e).removeprefix("Error: "))
            sys.exit(0)
        click.echo("Error: %s" % str(e).removeprefix("Error: "), err=True)
        sys.exit(e.exit_code if e.exit_code is not None else 1)
    except NoResumableState as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)
    except SparkrunError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)


def _run_benchmark(
    ctx,
    recipe_name,
    hosts,
    hosts_file,
    cluster_name,
    tensor_parallel,
    pipeline_parallel,
    data_parallel,
    gpu_mem,
    max_model_len,
    options,
    image,
    solo,
    port,
    profile,
    framework,
    output_file,
    bench_options,
    api_key_env,
    exit_on_first_fail,
    no_stop,
    skip_run,
    sync_tuning,
    rootful,
    bench_timeout,
    dry_run,
    executor_args,
    extra_args,
    export_results_files=True,
    fresh: bool = False,
    resume_mode: "ResumeMode | None" = None,
    on_prompt_required: "Callable[[Any], bool] | None" = None,
    submission_id_for_extras: str | None = None,
    scheduler_name: str | None = None,
    host_list=None,
    cluster_mgr=None,
    category: str | None = None,
):
    """Thin CLI presentation shell over ``sparkrun.api._benchmark._execute_benchmark``.

    Translates Click-shaped flags into ``BenchmarkOptions``, sets up the CLI
    progress emitter, calls the orchestration, and translates typed exceptions
    back into ``click.echo`` + ``sys.exit``.

    Returns the internal ``sparkrun.benchmarking.base.BenchmarkResult`` so
    existing callers (``_arena.py``, tests) don't break.
    """
    from sparkrun.api._benchmark_models import ResumeMode as _ResumeMode, BenchmarkOptions
    from sparkrun.api._benchmark import _execute_benchmark, _ProgressEmitter
    from sparkrun.api._errors import (
        BenchmarkFailed,
        NoResumableState,
        SparkrunError,
    )

    sctx = _get_context(ctx)

    # Translate legacy ``fresh`` bool to the new ResumeMode axis when caller
    # didn't provide one.
    if resume_mode is None:
        resume_mode = _ResumeMode.FRESH if fresh else _ResumeMode.AUTO

    # AUTO-mode resume prompting is a CLI/console concern.  The API
    # orchestration consults ``options.on_prompt_required`` instead of
    # importing console code, so supply a callback that renders the
    # interactive ``click.confirm`` prompt (or the non-TTY default) via
    # ``_resolve_resume_prompt``.  ``state.schedule`` carries the full
    # task count for the resumed run.
    if on_prompt_required is None:
        on_prompt_required = lambda state: _resolve_resume_prompt(state, len(state.schedule), None)  # noqa: E731

    # Parse bench_options key=value strings into a dict for BenchmarkOptions.
    # Validation (insecure api_key, malformed) happens in _execute_benchmark.
    bench_args_dict: dict = {}
    for opt_str in bench_options or ():
        if "=" not in opt_str:
            click.echo("Error: --bench-option must be key=value, got: %s" % opt_str, err=True)
            sys.exit(1)
        key, _, val = opt_str.partition("=")
        stripped_key = key.strip()
        if "api_key" in stripped_key.lower():
            click.echo(
                "Error: Passing '%s' via --bench-option is insecure. "
                "Please use the --api-key-env flag to pass it via an environment variable instead." % stripped_key,
                err=True,
            )
            sys.exit(1)
        bench_args_dict[stripped_key] = val.strip()

    # Build overrides dict from CLI recipe-override flags so _execute_benchmark
    # receives them via options.overrides.
    _dummy_recipe = None
    _overrides_from_flags: dict = {}
    if image:
        _overrides_from_flags["image"] = image
    if port:
        _overrides_from_flags["port"] = port
    # Handle the options tuple (tensor_parallel, pipeline_parallel, etc.) by
    # running _apply_recipe_overrides if there are any recipe-override args.
    # We need a recipe object for this — but we delay recipe loading to the API.
    # Instead, thread the raw CLI overrides through options.overrides so the API
    # can apply them itself.
    if tensor_parallel is not None:
        _overrides_from_flags["tensor_parallel"] = tensor_parallel
    if pipeline_parallel is not None:
        _overrides_from_flags["pipeline_parallel"] = pipeline_parallel
    if data_parallel is not None:
        _overrides_from_flags["data_parallel"] = data_parallel
    if gpu_mem is not None:
        _overrides_from_flags["gpu_mem"] = gpu_mem
    if max_model_len is not None:
        _overrides_from_flags["max_model_len"] = max_model_len
    # Apply options tuple (list of key=value strings from --option/-o flags)
    for opt_str in options or ():
        if "=" in opt_str:
            k2, _, v2 = opt_str.partition("=")
            _overrides_from_flags[k2.strip()] = v2.strip()

    state_extras: dict = {}
    if submission_id_for_extras:
        state_extras["submission_id"] = submission_id_for_extras

    opts = BenchmarkOptions(
        recipe=recipe_name,
        category=category,
        framework=framework,
        profile=profile,
        bench_args=bench_args_dict,
        hosts=tuple(host_list) if host_list else (tuple(hosts) if hosts else ()),
        cluster=cluster_name,
        overrides=_overrides_from_flags,
        resume=resume_mode,
        skip_run=skip_run,
        no_stop=no_stop,
        exit_on_first_fail=exit_on_first_fail,
        timeout=bench_timeout,
        api_key_env=api_key_env,
        arena=False,
        output_file=output_file,
        export_files=export_results_files,
        solo=solo,
        dry_run=dry_run,
        scheduler=scheduler_name,
        rootful=rootful,
        sync_tuning=bool(sync_tuning),
        extra_docker_opts=tuple(executor_args) if executor_args else None,
        progress_callback=None,
        state_extras=state_extras,
        on_prompt_required=on_prompt_required,
    )

    class _CliEmitter(_ProgressEmitter):
        """CLI emitter: prints banners/info/warnings/errors via click.echo."""

        def banner(self, line: str) -> None:
            click.echo(line)

        def info(self, msg: str) -> None:
            click.echo(msg)

        def warning(self, msg: str) -> None:
            click.echo("Warning: %s" % msg, err=True)

        def error(self, msg: str) -> None:
            click.echo("Error: %s" % msg, err=True)

        def progress_step(self, step_idx: int, total: int, label: str) -> None:
            pass  # CLI uses logger.log(PROGRESS_LEVEL, ...) inside orchestration

        def event(self, ev) -> None:
            pass  # BenchmarkProgressUI handles scheduled-task progress

        def on_recipe_resolved(self, recipe, overrides, *, local_cache_dir=None):
            # CLI-side presentation: VRAM estimate using the recipe loaded
            # once by the orchestration. Non-fatal on any failure.
            try:
                _display_vram_estimate(recipe, cli_overrides=overrides, auto_detect=True, cache_dir=local_cache_dir)
            except Exception:
                pass

    emitter = _CliEmitter()

    try:
        bench_result = _execute_benchmark(opts, sctx=sctx, emitter=emitter)
    except KeyboardInterrupt:
        click.echo("")
        click.echo("Interrupted.")
        sys.exit(130)
    except BenchmarkFailed as e:
        if e.exit_code is not None:
            sys.exit(e.exit_code)
        sys.exit(1)
    except NoResumableState as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)
    except SparkrunError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    return bench_result


def _stop_inference(runtime, host_list, cluster_id, config, dry_run, sctx=None):
    """Thin CLI shell over ``sparkrun.api._benchmark._stop_inference``.

    ``runtime`` is retained in the signature for backward compatibility with
    existing callers; the actual stop is dispatched through ``api.stop`` (via
    the canonical API helper) so executor / collective-backend lookup mirrors
    the launch path.  A click-backed emitter surfaces the dry-run notice and
    any warning to the console.
    """
    from sparkrun.api._benchmark import _stop_inference as _api_stop_inference, _ProgressEmitter

    class _CliEmitter(_ProgressEmitter):
        def info(self, msg: str) -> None:
            click.echo(msg)

        def warning(self, msg: str) -> None:
            click.echo("Warning: %s" % msg, err=True)

    _api_stop_inference(runtime, host_list, cluster_id, config, dry_run, sctx=sctx, emitter=_CliEmitter())
