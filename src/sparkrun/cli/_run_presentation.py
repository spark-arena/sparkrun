"""CLI presentation shared by direct and benchmark inference launches.

Render the actual RunPlan; never repeat placement or probe hardware for display.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import click

from sparkrun.core.application_profile import render_identity_text
from sparkrun.models.hub import hub_degraded_message

from ._common import _display_memory_plan

if TYPE_CHECKING:
    from sparkrun.api import RunOptions, RunPlan


def describe_host_source(*, hosts, hosts_file, cluster_name, cluster_mgr, config) -> str:
    """Name the CLI's host-selection source, before placement trims candidates."""
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
            if cluster_mgr.default_override() is not None:
                from sparkrun.core.cluster_manager import DEFAULT_CLUSTER_ENV

                host_source += " (from %s)" % DEFAULT_CLUSTER_ENV
        elif config.default_hosts:
            host_source = "config defaults"
        else:
            host_source = "localhost"

    return host_source


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


def _summarize_platforms(
    host_list: list[str],
    cluster=None,
) -> tuple[str, list[tuple[str, str]] | None]:
    """Build a platform summary string for the ``sparkrun run`` output block.

    For each host, resolves hardware (from *cluster* if available, else
    :func:`~sparkrun.core.hardware.resolve_hardware`), picks the
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
    from sparkrun.core.hardware import resolve_hardware
    from sparkrun import platforms as _platforms

    def _host_line(host: str) -> str:
        try:
            hw = cluster.hardware_for(host) if cluster is not None else resolve_hardware()
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

    return "mixed", list(zip(host_list, lines, strict=True))


def echo_run_identity(recipe, runtime) -> None:
    """Identify the workload before planning's potentially slow discovery."""
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.version import display_version

    click.echo(render_identity_text("{app_command} v%s" % display_version(SparkrunConfig())))
    click.echo()
    click.echo("Runtime:   %s" % runtime.runtime_name)
    click.echo("Image:     %s" % (recipe.container or "selected per host during image preparation"))
    click.echo("Model:     %s" % recipe.model)


@contextmanager
def planning_output():
    """Make discovery visible, including a clean line ending on failure."""
    click.echo("Planning:  probing hardware, querying cluster occupancy and model metadata...", nl=False)
    try:
        yield
    except BaseException:
        click.echo()
        raise
    else:
        click.echo(" done")
        _echo_hub_notice()


def render_run_plan(run_plan: RunPlan, options: RunOptions, *, host_source: str, verbose: bool = False) -> bool:
    """Display placement, hardware and memory; acknowledge reported hardware.

    The caller may mark the plan's hardware_reported flag only after this
    returns True, preventing duplicate evidence during launch. API callers
    without a terminal renderer retain the ordinary launch feedback.
    """
    recipe = run_plan.recipe
    host_list = list(run_plan.host_list)
    is_solo = run_plan.is_solo
    from sparkrun.core.scheduler import default_scheduler_upgrade_hint

    click.echo()
    if is_solo:
        click.echo("Placement: solo on %s (%s)" % (host_list[0] if host_list else "localhost", host_source))
    else:
        _of = " of %d" % len(run_plan.candidate_hosts) if len(run_plan.candidate_hosts) > len(host_list) else ""
        click.echo("Placement: %d%s hosts from %s (%s scheduler)" % (len(host_list), _of, host_source, run_plan.scheduler))
    for _note in run_plan.notes:
        click.echo(_note)
    if recipe.mode == "cluster" and is_solo and not options.solo:
        click.echo("Warning: Recipe requires cluster mode but only one host specified", err=True)
    if run_plan.scheduler_defaulted and not is_solo:
        # Nothing in the chain selected a scheduler, so the 0.2.x greedy
        # default applied; recommend occupancy-aware spreading.
        click.echo(default_scheduler_upgrade_hint())
    if run_plan.offline is not None and (run_plan.offline.offline or run_plan.offline.source != "default"):
        click.echo("Offline:   %s" % run_plan.offline.describe())
    if options.transfer_mode not in ("auto", "local"):
        click.echo("Transfer:  %s" % options.transfer_mode)
    # Which conditional `overrides:` applied, so a value that differs from the
    # recipe's `defaults` is explained before the memory plan that uses it.
    if run_plan.override_resolution is not None:
        for _line in run_plan.override_resolution.describe():
            click.echo(_line)
    _platform_summary, _per_host = _summarize_platforms(host_list, run_plan.cluster)
    click.echo("Platform:  %s" % _platform_summary)
    if _per_host is not None:
        for _h, _line in _per_host:
            click.echo("  %-8s %s" % (_h + ":", _line))

    # What the plan probed, as one table, rather than as evidence strings inside
    # phase 1 after the fit they back had already been read.
    _assessment = run_plan.hardware_assessment
    if _assessment is not None and _assessment.hosts:
        from sparkrun.utils.cli_formatters import format_host_table

        for _line in format_host_table(
            _assessment, host_list, run_plan.placement, is_solo=is_solo, verbose=verbose, dry_run=options.dry_run
        ):
            click.echo(_line)
    else:
        for _role, _hosts in (("Target", host_list[:1]),) if is_solo else (("Head", host_list[:1]), ("Workers", host_list[1:])):
            if _hosts:
                click.echo("  %-8s %s" % (_role + ":", ", ".join(_hosts)))
    if _assessment is not None:
        for _warning in _assessment.warnings:
            click.echo("Warning: %s" % _warning, err=True)

    # The memory plan renders the plan's own placement.  It used to come from
    # a third scheduling call, deliberately made *without* live occupancy —
    # which is how a capacity failure could print a table showing every target
    # host as [OK] directly above the error that rejected them.
    #
    # A local (absolute-path) model has no HuggingFace repo id to auto-detect
    # params from, so skip the HF lookup — the estimate falls back to recipe
    # defaults rather than erroring on a bogus repo id.
    from sparkrun.core.recipe import is_local_model_path

    _display_memory_plan(
        recipe,
        cluster=run_plan.cluster,
        placement=run_plan.placement,
        cli_overrides=dict(options.overrides),
        auto_detect=not is_local_model_path(recipe.model),
        cache_dir=options.local_cache_dir,
    )
    _echo_hub_notice()
    click.echo()

    return _assessment is not None
