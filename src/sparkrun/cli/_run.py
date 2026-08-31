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
from sparkrun.core.validation import FAIL_ON_CHOICES, validate_for_launch
from sparkrun.models.hub import disable_hub_metadata, hub_degraded_message
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
    _render_capacity_diagnostics,
    recipe_override_options,
    report_launch_validation,
    resolve_cluster_config,
    with_host_context,
    HIDE_ADVANCED_OPTIONS,
)

logger = logging.getLogger(__name__)

#: ``-o key=value`` keys that configure the *executor* rather than the serve
#: command.  They are lifted out of ``overrides`` into ``executor_config``
#: before placement, so the intent id the CLI derives matches the one
#: ``api.run`` derives from the same (already-stripped) override dict.
_EXECUTOR_OVERRIDE_KEYS = frozenset(
    {
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
        # ``-o entrypoint=''`` clears a consuming image ENTRYPOINT (one that
        # parses sparkrun's appended ``bash -c`` as its own flags) without
        # having to fork a third-party recipe just to add two lines of
        # executor_config.  Empty string is the meaningful value here and
        # survives ``coerce_value``; see ExecutorConfig.entrypoint.
        "entrypoint",
    }
)


def _echo_hub_notice() -> None:
    """Report skipped HuggingFace Hub lookups, at most once per command.

    Called at both points where the advisory phase can run out of budget — the
    plan, and the VRAM table it renders afterwards — because either can be the
    one that trips the breaker, and the notice has to land next to the pause it
    explains.  ``hub_degraded_message`` returning the string only once is what
    makes calling it from both sites correct rather than merely tolerable.
    """
    notice = hub_degraded_message()
    if notice:
        click.echo()
        click.echo(notice, err=True)


def _echo_endpoint_ready(readiness) -> None:
    """Announce a now-serving endpoint from the readiness watcher thread.

    Called while ``docker logs -f`` is writing to the same terminal, so it
    must be **one short line in one write**: a multi-line block would be
    interleaved with log output mid-render.  The full breakdown waits for
    the finalize step, once the stream has stopped.

    Goes to stderr so a caller piping the log stream keeps it uncontaminated.
    """
    from sparkrun.utils.text import format_duration

    click.secho(
        "\n[sparkrun] Endpoint ready at http://%s:%d/v1 after %s (engine init %s, model load %s)\n"
        % (
            readiness.head_ip,
            readiness.port,
            format_duration(readiness.total_wait_s),
            format_duration(readiness.port_wait_s),
            format_duration(readiness.health_wait_s),
        ),
        fg="green",
        err=True,
    )


def _report_readiness_outcome(readiness) -> None:
    """Report a readiness watch that ended without the endpoint serving.

    Deliberately does **not** touch the exit code.  The watch is
    observational: it runs on every launch now, and a slow-loading model
    that outlasts the poll budget must not turn a successful launch into a
    failure for everything scripted around ``sparkrun run``.

    Silent for ``None`` (still polling when we exited) and for
    ``"cancelled"`` (the user stopped the stream) — neither says anything
    about the workload.
    """
    if readiness is None or readiness.ready or readiness.reason == "cancelled":
        return
    if readiness.reason == "port":
        detail = "the head container stopped or port %d never opened" % readiness.port
    else:
        detail = "%s never returned HTTP 200" % readiness.health_url
    click.secho(
        "[sparkrun] WARNING: endpoint did not become ready — %s." % detail,
        fg="yellow",
        err=True,
    )


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
@click.option(
    "--dashboard/--no-dashboard",
    "dashboard",
    default=None,
    help="Enable/disable the Ray dashboard on the head node (Ray runtimes only; binds 0.0.0.0 when on). "
    "Overrides the recipe's runtime_config.dashboard; defaults to on.",
    hidden=HIDE_ADVANCED_OPTIONS,
)
@click.option("--dashboard-port", type=int, default=8265, help="Ray dashboard port", hidden=HIDE_ADVANCED_OPTIONS)
@dry_run_option
@click.option("--foreground", is_flag=True, help="Run in foreground (don't detach)")
@click.option("--ensure", is_flag=True, default=False, help="Only launch if not already running; exit 0 if already up")
@click.option("--no-follow", is_flag=True, help="Don't follow container logs after launch")
@click.option(
    "--no-auto-detect",
    is_flag=True,
    help="Skip HuggingFace Hub metadata lookups (VRAM estimate falls back to recipe metadata)",
)
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
    type=click.Choice(["auto", "local", "push", "delegated", "pull"], case_sensitive=False),
    help=(
        "Resource transfer mode (overrides cluster setting). 'pull' has every node fetch from "
        "origin itself in parallel — needs registry/HF credentials on each node, costs N x egress, "
        "and does not fall back to the control machine (use 'auto' for that). Note a re-pushed "
        "image tag is not re-pulled unless you also pass --rebuild."
    ),
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
    "--timings/--no-timings",
    "show_timings",
    default=True,
    hidden=HIDE_ADVANCED_OPTIONS,
    help="Print a per-stage timing breakdown when the run finishes (on by default; --no-timings suppresses it)",
)
@click.option(
    "--trust", is_flag=True, default=False, hidden=True, help="Trust post_commands from third-party registries without confirmation"
)
@click.option(
    "--runtime-cache/--no-runtime-cache",
    "runtime_cache",
    default=None,
    hidden=HIDE_ADVANCED_OPTIONS,
    help=(
        "Persist compilation/autotune caches (torch.compile, Triton, FlashInfer, TRT-LLM "
        "autotuner) on the target hosts across launches. Defaults to the recipe/cluster/config setting."
    ),
)
@click.option(
    "--scheduler",
    "scheduler_name",
    default=None,
    help="Registered scheduler name (e.g. 'greedy', 'occupancy-sparse', 'occupancy-dense'). Defaults to the recipe's scheduler field, then 'greedy'.",
    hidden=HIDE_ADVANCED_OPTIONS,
)
@click.option(
    "--rebuild/--no-rebuild",
    "rebuild",
    default=None,
    help="Force a fresh image: an unconditional 'docker pull' for registry images (the default docker-pull path), "
    "a from-scratch rebuild for eugr. Use when a local copy is stale or incomplete. "
    "Overrides the recipe's builder_config.rebuild setting.",
    hidden=HIDE_ADVANCED_OPTIONS,
)
@click.option(
    "--env",
    "-e",
    "env_overrides",
    multiple=True,
    hidden=HIDE_ADVANCED_OPTIONS,
    help="Set a container environment variable: -e KEY=VALUE (repeatable). Value is used verbatim, unlike -o env.KEY=VALUE.",
)
@click.option("--label", "labels_override", multiple=True, help="Set meta data on a container (e.g., --label com.example.key=value)")
@click.option(
    "--executor-args",
    multiple=True,
    hidden=HIDE_ADVANCED_OPTIONS,
    help="Arguments passed directly to the container executor (e.g. docker run)",
)
@click.option(
    "--fail-on",
    type=click.Choice(FAIL_ON_CHOICES),
    default=None,
    hidden=True,
    help="Least-severe recipe-validation finding that refuses the launch (default: error)",
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
    no_auto_detect,
    no_sync_tuning,
    no_rm,
    memory,
    rootful,
    restart_policy,
    transfer_mode,
    diagnostics_path,
    show_timings,
    trust,
    runtime_cache,
    scheduler_name,
    rebuild,
    env_overrides,
    labels_override,
    options,
    executor_args,
    fail_on,
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

      sparkrun run my-recipe.yaml -e VLLM_USE_V1=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    """
    from sparkrun.core.bootstrap import get_runtime

    sctx = _get_context(ctx)
    v = sctx.variables
    config = sctx.config

    # Set before anything can reach the Hub.  Process-wide rather than threaded
    # down as a parameter: ``estimate_vram`` is called from host resolution, the
    # banner, the scheduling pass inside ``api.run`` and telemetry, and a flag
    # that reached three of those four would still hang on the fourth.
    if no_auto_detect:
        disable_hub_metadata()

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

    # -e/--env lands in the same place as -o env.KEY=VALUE (recipe.env, the top
    # container-env tier) but keeps the value verbatim. Applied after, so it
    # wins if a key is given both ways.
    if env_overrides:
        from sparkrun.core.resolve import apply_env_overrides

        try:
            apply_env_overrides(recipe, env_overrides)
        except ValueError as e:
            raise click.UsageError(str(e)) from e

    # --rebuild/--no-rebuild is a builder-agnostic override carried in
    # builder_config so any builder (present or future) can honor it. Only
    # override the recipe's own builder_config.rebuild when the flag was given
    # explicitly (tri-state default None leaves the recipe value untouched).
    if rebuild is not None:
        recipe.builder_config["rebuild"] = rebuild

    # Get assigned runtime
    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    # Validate recipe (after resolve so runtime is populated).  Errors abort
    # before any side effect: they name something sparkrun cannot honor (an
    # unresolvable builder or executor, a runtime that rejects the recipe), and
    # launching anyway produces a deployment that isn't the one described.
    # Warnings print and continue — each is a supported escape hatch.
    #
    # The unmapped-key half is left off here: ``launch_inference`` runs it
    # after the platform default tier and with ``-o`` already split into serve
    # vs executor keys, so running it now would both duplicate the output and
    # report executor overrides (``-o entrypoint=''``) as having no effect.
    issues, validation_failed = validate_for_launch(
        recipe,
        fail_on=fail_on,
        runtime=runtime,
        cluster=cluster_def,
        config=config,
        v=v,
        include_unmapped_keys=False,
        # Echo back the reference the user typed: it is what the `recipe
        # validate` hint has to be re-typable as, and for a URL recipe it beats
        # the raw URL. Threaded into validation as well as into the report,
        # because the collapsed deprecation line names the same command.
        recipe_ref=recipe_name,
    )
    report_launch_validation(recipe_name, issues, validation_failed)
    if validation_failed:
        sys.exit(1)

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

    # Extract executor-specific keys from -o/--option overrides.  Done before
    # the RunOptions build below so ``overrides`` and ``executor_config`` are
    # already separated when the plan is computed — the plan's intent id and
    # config chain must be the ones the launch uses.
    option_executor_opts: dict[str, Any] = {}
    for key in list(overrides.keys()):
        if key in _EXECUTOR_OVERRIDE_KEYS:
            option_executor_opts[key] = overrides.pop(key)

    # Resolve cache dir, transfer mode, and transfer interface from cluster
    # config.  Independent of placement, so it can precede the plan.
    cluster_cfg = resolve_cluster_config(cluster_name, hosts, hosts_file, cluster_mgr)
    local_cache_dir, remote_cache_dir, effective_transfer_mode, effective_transfer_interface = cluster_cfg.resolve_transfer_config(
        config, transfer_mode_override=transfer_mode
    )

    # Build executor config from CLI flags, then layer the -o/--option-sourced
    # keys on top — preserving the precedence where -o wins over the flag.
    cli_executor_opts: dict[str, Any] = {}
    if no_rm:
        cli_executor_opts["auto_remove"] = False
    if memory:
        cli_executor_opts["memory_limit"] = memory
    if restart_policy:
        cli_executor_opts["restart_policy"] = restart_policy
    if labels_override:
        cli_executor_opts["labels"] = list(labels_override)
    cli_executor_opts.update(option_executor_opts)

    # Build the typed RunOptions for the library API.  The CLI already
    # resolved the recipe, host list, cluster_def, and overrides above (so the
    # banner / VRAM block can render those before launch); passing the loaded
    # objects through avoids re-resolution inside the API and preserves the
    # cwd-recipe discovery the CLI does through ``_load_recipe``.
    #
    # ``hosts`` is the **candidate** list — every host placement may choose
    # from — not a pre-trimmed selection.  Narrowing it here would make this
    # command's own placement authoritative over the launch's, leaving
    # ``api.run`` unable to reach any host we discarded.
    run_options = api.RunOptions(
        recipe=recipe,
        hosts=tuple(host_list),
        cluster=cluster_def,
        overrides=dict(overrides),
        scheduler=scheduler_name,
        solo=solo,
        dry_run=dry_run,
        follow=not no_follow,
        detached=not foreground,
        trust=trust,
        transfer_mode=effective_transfer_mode,
        transfer_interface=effective_transfer_interface,
        cache_dir=remote_cache_dir,
        runtime_cache=runtime_cache,
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

    # --ensure: if this workload is already serving, don't launch — and don't
    # schedule either, which is why this runs before the plan.
    #
    # Keyed on the *intent* (recipe + parallelism + port), never on a
    # host-derived cluster_id: a cluster_id also encodes placement, so the old
    # lookup could not match a job placed by a status-aware scheduler (random
    # placement token) and would launch a duplicate every time.  The intent is
    # placement-independent, so the answer no longer depends on which scheduler
    # is configured.  Hosts are the full candidate list — a deployment that
    # landed on hosts this launch wouldn't pick still counts as running.
    if ensure:
        from sparkrun.orchestration.job_metadata import generate_intent_id

        _match = api.find_running_intent(
            generate_intent_id(recipe, overrides),
            host_list,
            cluster=cluster_def,
            sctx=sctx,
        )
        if _match is not None:
            click.echo("Job already running (cluster_id: %s)" % _match.cluster_id)
            click.echo("  Recipe: %s" % (_match.recipe or recipe.qualified_name or "unknown"))
            click.echo("  Hosts:  %s" % ", ".join(_match.hosts))
            if _match.other_cluster_ids:
                click.echo(
                    "  Warning: %d other deployment(s) of this workload are also running: %s"
                    % (len(_match.other_cluster_ids), ", ".join(_match.other_cluster_ids)),
                    err=True,
                )
            sys.exit(0)

    # Identity half of the banner, printed *before* the plan.
    #
    # Planning sweeps the cluster over SSH and resolves model metadata from the
    # HuggingFace Hub, which on a slow or rate-limited Hub is the longest pause
    # in the whole command — and used to be a completely silent one, with the
    # first byte of output arriving only after it finished (issue #278).  These
    # four lines need nothing from the plan, so printing them here makes the
    # pause attributable without making the display disagree with the launch:
    # everything placement-dependent still renders from ``run_plan`` below.
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.version import display_version

    container_image = runtime.resolve_container(recipe, overrides)
    click.echo("sparkrun v%s" % display_version(SparkrunConfig()))
    click.echo()
    click.echo("Runtime:   %s" % runtime.runtime_name)
    click.echo("Image:     %s" % container_image)
    click.echo("Model:     %s" % recipe.model)
    click.echo("Planning:  querying cluster occupancy and model metadata...", nl=False)

    # Decide the launch ONCE.  Everything below renders from this plan, and the
    # plan is handed back to ``api.run`` — so what is displayed is exactly what
    # is launched.  Scheduling here and passing the winners as ``hosts`` instead
    # would re-narrow the candidate set and reintroduce the class of failure
    # where a launch reports "insufficient free capacity" while idle hosts sit
    # unused because this pass already discarded them.
    try:
        run_plan = api.plan(run_options, sctx=sctx)
    except api.InsufficientCapacity as e:
        click.echo()
        click.echo("Error: %s" % e, err=True)
        _render_capacity_diagnostics(getattr(e, "status", None), list(getattr(e, "host_list", ()) or host_list))
        sys.exit(1)
    except api.SparkrunError as e:
        click.echo()
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)
    click.echo(" done")

    _echo_hub_notice()

    for _note in run_plan.notes:
        click.echo(_note)

    host_list = list(run_plan.host_list)
    is_solo = run_plan.is_solo
    if recipe.mode == "cluster" and is_solo and not solo:
        click.echo("Warning: Recipe requires cluster mode but only one host specified", err=True)

    # Placement-dependent half of the summary; the identity half printed above
    # the plan.
    from sparkrun.core.scheduler import default_scheduler_upgrade_hint

    if is_solo:
        click.echo("Mode:      solo")
    else:
        click.echo("Mode:      cluster (%d nodes)" % len(host_list))
    _platform_summary, _per_host = _summarize_platforms(host_list, run_plan.cluster)
    click.echo("Platform:  %s" % _platform_summary)
    if _per_host is not None:
        for _h, _line in _per_host:
            click.echo("  %-8s %s" % (_h + ":", _line))
    click.echo("Scheduler: %s" % run_plan.scheduler)
    # When nothing in the chain selected a scheduler we fell back to the 0.2.x
    # greedy default; recommend opting the cluster into occupancy-aware spreading.
    if run_plan.scheduler_defaulted and not is_solo:
        click.echo(default_scheduler_upgrade_hint())
    if effective_transfer_mode not in ("auto", "local"):
        click.echo("Transfer:  %s" % effective_transfer_mode)

    # The per-host fit table renders the plan's own placement.  It used to be
    # a third scheduling call, deliberately made *without* live occupancy —
    # which is how a capacity failure could print a table showing every target
    # host as [OK] directly above the error that rejected them.
    #
    # A local (absolute-path) model has no HuggingFace repo id to auto-detect
    # params from, so skip the HF lookup — the estimate falls back to recipe
    # defaults rather than erroring on a bogus repo id.
    from sparkrun.core.recipe import is_local_model_path

    _display_vram_estimate(
        recipe,
        cli_overrides=overrides,
        auto_detect=not is_local_model_path(recipe.model),
        cache_dir=local_cache_dir,
        cluster=run_plan.cluster,
        placement=run_plan.placement,
    )
    _echo_hub_notice()

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

    # Own the timeline here rather than letting ``launch_inference`` create one
    # internally: on a failed launch there is no LaunchResult to read it off,
    # and a launch that failed is exactly when the per-stage breakdown is worth
    # having.  ``launch_inference`` picks this up via ``sctx``.
    #
    # ``--no-timings`` suppresses the *table*, not the readiness watch below —
    # "the endpoint is up now" is worth having while logs scroll whether or
    # not you want a breakdown afterwards.  Diagnostics needs the timeline for
    # its own record regardless.
    if show_timings or diagnostics_path:
        from sparkrun.core.timing import Timeline

        sctx.timing = Timeline()

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

    # Launch via the library API; the API call internally drives
    # ``launch_inference`` (which calls ``runtime.run``).  Tests that
    # mock ``runtime.run`` still observe the call because the runtime
    # layer is unchanged.
    #
    # ``plan=run_plan`` launches exactly what was displayed above — no
    # second resolution, no second transport prepare, no second occupancy
    # sweep, and no chance of the launch landing somewhere other than the
    # hosts named in the banner.
    if diag:
        diag.phase_start("launch")
    # Spans the whole API call, not just ``launch_inference``: the difference
    # between this and the ``launch`` span nested inside it is ``api.run``'s
    # own preamble (transport prepare, eviction setup), which is otherwise
    # unattributed time the reader can see in the total but not account for.
    # Left open on the error paths below, which is what ``status="open"``
    # means — the run did not finish.
    _run_span = sctx.timing.begin("run") if sctx.timing is not None else None
    try:
        run_result = api.run(run_options, sctx=sctx, plan=run_plan)
    except TransferError as e:
        if diag:
            diag.phase_end("launch", error=str(e))
            diag.emit_error("launch", e)
            diag.emit_timeline(sctx.timing)
            diag.emit_summary()
            diag.close()
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)
    except IncompatibleHardwareError as e:
        if diag:
            diag.phase_end("launch", error=str(e))
            diag.emit_error("launch", e)
            diag.emit_timeline(sctx.timing)
            diag.emit_summary()
            diag.close()
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)
    except api.SparkrunError as e:
        if diag:
            diag.phase_end("launch", error=str(e))
            diag.emit_error("launch", e)
            diag.emit_timeline(sctx.timing)
            diag.emit_summary()
            diag.close()
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)
    except Exception as e:
        if diag:
            diag.phase_end("launch", error=str(e))
            diag.emit_error("launch", e)
            diag.emit_timeline(sctx.timing)
            diag.emit_summary()
            diag.close()
        raise

    if _run_span is not None:
        sctx.timing.end(_run_span, rc=int(run_result.rc))

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
    watcher = None
    serving_span = None
    if result.rc == 0 and not foreground and not dry_run:
        if not no_follow:
            # The readiness poll runs *alongside* the log stream rather than
            # before it.  ``launch_inference`` returns once the containers are
            # up, which for a large model is minutes before the server accepts
            # a request — and those minutes are precisely what the user is
            # watching scroll past.  Waiting first would blank the screen for
            # the most informative part of the launch; not waiting at all is
            # what left `serve.port_open` / `serve.health_ok` unrecorded on
            # every run of a recipe without post hooks.
            #
            # Skipped when the recipe *has* post hooks: `post_launch_lifecycle`
            # above already waited synchronously and recorded those spans, so a
            # watcher here would duplicate them and re-poll a live endpoint.
            if not has_post_hooks:
                from sparkrun.core.launcher import ReadinessWatcher
                from sparkrun.orchestration.primitives import build_ssh_kwargs as _watch_ssh

                watcher = ReadinessWatcher(
                    result,
                    ssh_kwargs=_watch_ssh(config),
                    on_ready=_echo_endpoint_ready,
                    timeline=sctx.timing,
                ).start()
            elif sctx.timing is not None:
                # The post-hook path waited synchronously above, so the
                # endpoint is already serving and there is no watcher to own
                # the interval — but the log stream below still runs for as
                # long as the user watches, and that time is just as
                # unaccounted here as it would be there.
                from sparkrun.core.timing import ROOT as _TIMELINE_ROOT

                serving_span = sctx.timing.begin("serve.serving", parent=_TIMELINE_ROOT, label="serving")

            # `finally`, not a plain follow-up statement: the watcher holds a
            # thread that polls over SSH, and it must be cancelled even if the
            # stream ends by an exception rather than by the user.
            try:
                runtime.follow_logs(
                    hosts=host_list,
                    cluster_id=result.cluster_id,
                    config=config,
                    dry_run=dry_run,
                )
            finally:
                # ``watcher.stop()`` closes its own serving span; this one is
                # the post-hook path's, which has no watcher.
                readiness = watcher.stop() if watcher is not None else None
                if serving_span is not None and sctx.timing is not None:
                    sctx.timing.end(serving_span)

            # Reached when the user interrupts the stream (Ctrl-C is caught
            # inside the log printer) or the container exits and `docker logs
            # -f` ends.  Either way nothing else owns the terminal from here.
            _report_readiness_outcome(readiness)
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

    # Printed last, and only here.  The table is multi-line, so it cannot be
    # emitted while `docker logs -f` is writing to the same terminal without
    # being shredded mid-render — the live half of the report is the single
    # line `_echo_endpoint_ready` injects.  By this point the stream has
    # stopped and nothing else is writing.
    #
    # A watcher still polling when the user interrupts leaves `serve.*` open,
    # which `format_launch_timings` renders "did not finish" — the honest
    # reading of "we stopped watching", not a claim that the stage failed.
    if show_timings and sctx.timing is not None:
        from sparkrun.utils.cli_formatters import format_launch_timings

        _timings = format_launch_timings(sctx.timing.export())
        if _timings:
            click.echo()
            click.echo(_timings)
            click.echo()

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
        diag.emit_timeline(sctx.timing)
        diag.emit_summary()
        diag.close()
        click.echo("Diagnostics written to: %s" % diagnostics_path)

    sys.exit(result.rc)
