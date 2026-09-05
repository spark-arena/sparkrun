"""``sparkrun setup throttle-gpu-clock`` — report or cap the GPU SM clock on cluster hosts."""

from __future__ import annotations

import sys

import click

from .._common import (
    HIDE_ADVANCED_OPTIONS,
    _resolve_setup_context,
    dry_run_option,
    host_options,
)
from . import setup

#: SM clock ceiling (MHz) suggested by the status report.  Not applied
#: implicitly — locking the clock always takes an explicit MAX_CLOCK, since a
#: bare invocation reports rather than changes.
SUGGESTED_MAX_CLOCK_MHZ = 2000

#: Sanity bounds for the MAX_CLOCK argument — a value outside these is far more
#: likely a unit mistake (GHz, or a stray digit) than a clock anyone wants.
MIN_MAX_CLOCK_MHZ = 100
MAX_MAX_CLOCK_MHZ = 10000

#: The boot-time unit ``--persistent`` installs.  A clock lock is driver state
#: that a reboot discards, so surviving one means reapplying it at boot.
UNIT_NAME = "sparkrun-gpu-clock.service"
UNIT_PATH = "/etc/systemd/system/" + UNIT_NAME

_STATUS_TIMEOUT_S = 30
_APPLY_TIMEOUT_S = 60


def _gpu_clock_summary(stdout: str) -> str:
    """Extract the status lines from the throttle script's output."""
    keywords = ("LOCKED:", "CLEARED:", "UNIT:", "CURRENT:", "ERROR:")
    lines = [line.strip() for line in stdout.strip().splitlines() if any(line.strip().startswith(kw) for kw in keywords)]
    return "; ".join(lines) if lines else stdout.strip()[:100]


def _parse_status_rows(stdout: str) -> list[tuple[str, str, str, str, str]]:
    """Parse the per-GPU CSV emitted by ``gpu_clock_status.sh``.

    Rows are ``(index, current_sm, max_sm, applications_graphics, persistence)``.
    nvidia-smi reports unavailable fields as ``[N/A]`` / ``[Not Supported]``;
    those become ``-`` rather than being guessed at.
    """
    rows = []
    for line in stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5 or not parts[0].isdigit():
            continue
        rows.append(tuple("-" if p.startswith("[") else p for p in parts))  # type: ignore[arg-type]
    return rows


def _parse_unit_state(stdout: str) -> str:
    """Render the ``UNIT:`` line from the status script as a table cell.

    ``UNIT: none`` → ``-``; ``UNIT: 0,2150 enabled`` → ``0,2150``; an installed
    but not-enabled unit keeps its state, since it would *not* reapply at boot.
    """
    for line in stdout.strip().splitlines():
        if not line.startswith("UNIT:"):
            continue
        parts = line[len("UNIT:") :].split()
        if not parts or parts[0] == "none":
            return "-"
        spec = parts[0]
        state = parts[1] if len(parts) > 1 else "unknown"
        return spec if state == "enabled" else "%s (%s)" % (spec, state)
    return "-"


def _mhz(value: str) -> str:
    return "%s MHz" % value if value.isdigit() else value


def _report_status(host_list, ssh_kwargs, dry_run) -> int:
    """Print the current per-GPU clock state. Returns the failed-host count."""
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel
    from sparkrun.scripts import read_script

    script = read_script("gpu_clock_status.sh")
    results = run_remote_scripts_parallel(
        host_list,
        script,
        timeout=_STATUS_TIMEOUT_S,
        quiet=True,
        dry_run=dry_run,
        **ssh_kwargs,
    )

    header = ("HOST", "GPU", "SM CLOCK", "MAX SM", "APPLICATIONS", "PERSISTENCE*", "AT BOOT")
    table = [header]
    failures = []
    for r in results:
        if not r.success:
            failures.append((r.host, (r.stderr or r.stdout).strip()[:200]))
            continue
        rows = _parse_status_rows(r.stdout)
        if not rows:
            failures.append((r.host, (r.stdout or r.stderr).strip()[:200] or "no GPUs reported"))
            continue
        unit = _parse_unit_state(r.stdout)
        for idx, cur, mx, apps, persistence in rows:
            table.append((r.host, idx, _mhz(cur), _mhz(mx), _mhz(apps), persistence, unit))

    if len(table) > 1:
        widths = [max(len(row[i]) for row in table) for i in range(len(header))]
        for row in table:
            click.echo("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip())
        click.echo()
        click.echo("* driver stays loaded while idle — holds the lock, but not across reboots")
        click.echo()
        # The driver has no read-back for --lock-gpu-clocks: NVML exports the
        # setter (nvmlDeviceSetGpuLockedClocks) but no getter, and nvidia-smi
        # has no matching --query-gpu field.  AT BOOT is sparkrun's own unit,
        # which is readable; the live lock is only visible by its effect.
        click.echo("Note: the driver does not report the --lock-gpu-clocks setting itself.")
        click.echo("      A lock shows up as SM CLOCK capped below MAX SM while the GPU is busy.")
        click.echo("      AT BOOT is the %s unit, which reapplies a lock after reboot." % UNIT_NAME)
        click.echo()
        click.echo("Cap the clock:  sparkrun setup throttle-gpu-clock %d" % SUGGESTED_MAX_CLOCK_MHZ)
        click.echo("Remove a cap:   sparkrun setup throttle-gpu-clock --clear")

    for host, message in failures:
        click.echo("  [FAIL] %s: %s" % (host, message), err=True)

    return len(failures)


def _confirm_boot_unit(host_list, max_clock, yes, dry_run) -> bool:
    """Second gate, in front of the boot-time unit only.

    Installing it writes to /etc/systemd/system and outlives the session, so it
    is confirmed separately from the clock lock itself.  Declining keeps the
    runtime lock — the first prompt already covered that.
    """
    click.echo("--persistent also installs a systemd unit on each of the %d host(s):" % len(host_list))
    click.echo("  %s" % UNIT_PATH)
    click.echo("  ExecStart=<nvidia-smi> --lock-gpu-clocks 0,%d" % max_clock)
    click.echo("  enabled at boot, ordered after nvidia-persistenced.service")
    click.echo("It persists until removed with --clear.")
    click.echo()
    if dry_run or yes:
        return True
    if click.confirm("Install the boot-time unit?", default=False):
        return True
    click.echo("Skipping the boot-time unit; applying the runtime lock only.")
    return False


@setup.command("throttle-gpu-clock", hidden=HIDE_ADVANCED_OPTIONS)
@click.argument("max_clock", type=int, required=False, default=None)
@host_options
@click.option("--user", "-u", default=None, help="SSH username (default: from config or current user)")
@click.option("--clear", "clear_lock", is_flag=True, help="Remove the clock lock and any boot-time unit (same as MAX_CLOCK 0)")
@click.option("--persistent", is_flag=True, help="Also install a systemd unit that reapplies the lock at boot")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompts (including the boot-time unit)")
@dry_run_option
@click.pass_context
def setup_throttle_gpu_clock(ctx, max_clock, hosts, hosts_file, cluster_name, user, clear_lock, persistent, yes, dry_run):
    """Report or cap the GPU SM clock on cluster hosts (nvidia-smi --lock-gpu-clocks).

    With no MAX_CLOCK and no --clear, this only *reports* the current per-GPU
    clock state and changes nothing.

    Given a MAX_CLOCK, locks each host's GPU to a ``0,MAX_CLOCK`` MHz range,
    which caps power draw and heat at the cost of peak throughput.  2000 is a
    reasonable starting point on DGX Spark.

    Passing ``0`` or ``--clear`` removes the lock (``--reset-gpu-clocks``) and
    any boot-time unit sparkrun installed.

    A lock is driver state: it does not survive a reboot, and it is dropped
    when the driver unloads (which on Linux happens once the last client
    exits, unless persistence mode is on).  ``--persistent`` therefore
    installs a systemd oneshot that reapplies it at boot — a durable change to
    each host, confirmed separately.

    Reporting needs only SSH; changing the lock requires sudo on target hosts.

    Examples:

      sparkrun setup throttle-gpu-clock --cluster mylab

      sparkrun setup throttle-gpu-clock 2150 --cluster mylab

      sparkrun setup throttle-gpu-clock 2150 --persistent --cluster mylab

      sparkrun setup throttle-gpu-clock --clear --cluster mylab
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.sudo import run_with_sudo_fallback, run_sudo_script_on_host
    from sparkrun.scripts import read_script

    if clear_lock and max_clock not in (None, 0):
        raise click.UsageError("--clear conflicts with MAX_CLOCK %d; pass one or the other." % max_clock)

    # No target clock and no --clear: report only, touch nothing.
    status_only = max_clock is None and not clear_lock
    reset = clear_lock or max_clock == 0

    if persistent and (status_only or reset):
        raise click.UsageError("--persistent needs a MAX_CLOCK to persist; --clear already removes the boot-time unit.")

    if not status_only and not reset and not (MIN_MAX_CLOCK_MHZ <= max_clock <= MAX_MAX_CLOCK_MHZ):
        raise click.BadParameter(
            "MAX_CLOCK must be 0 (clear) or between %d and %d MHz, got %d." % (MIN_MAX_CLOCK_MHZ, MAX_MAX_CLOCK_MHZ, max_clock),
            param_hint="MAX_CLOCK",
        )

    config = SparkrunConfig()
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)

    if status_only:
        click.echo("GPU clock state on %d host(s):" % len(host_list))
        click.echo()
        if _report_status(host_list, ssh_kwargs, dry_run):
            sys.exit(1)
        return

    if reset:
        action = "Clearing GPU clock lock (and any boot-time unit)"
    else:
        action = "Locking GPU clocks to 0,%d MHz" % max_clock
    click.echo("%s on %d host(s):" % (action, len(host_list)))
    for h in host_list:
        click.echo("  %s" % h)
    click.echo()

    if not dry_run and not yes and not click.confirm("Proceed?", default=False):
        click.echo("Aborted.")
        return

    # Second gate: the boot-time unit outlives the session, so it is confirmed
    # on its own.  Declining leaves the runtime lock the user already approved.
    install_unit = persistent and _confirm_boot_unit(host_list, max_clock, yes, dry_run)

    # "keep" leaves an existing unit alone and reports it, so a lock that
    # disagrees with what boots is visible rather than silent.
    unit_action = "install" if install_unit else ("remove" if reset else "keep")
    params = {
        "clock_args": "--reset-gpu-clocks" if reset else "--lock-gpu-clocks 0,%d" % max_clock,
        "result_label": "CLEARED: gpu clock lock removed" if reset else "LOCKED: 0,%d MHz" % max_clock,
        "unit_action": unit_action,
    }
    script = read_script("gpu_clock_throttle.sh").format(**params)
    fallback_script = read_script("gpu_clock_throttle_fallback.sh").format(**params)

    # Try non-interactive sudo, then password-based fallback
    result_map, still_failed = run_with_sudo_fallback(
        host_list,
        script,
        fallback_script,
        ssh_kwargs,
        dry_run=dry_run,
        timeout=_APPLY_TIMEOUT_S,
    )

    for h in host_list:
        r = result_map.get(h)
        if r and r.success:
            click.echo("  [OK]   %s: %s" % (h, _gpu_clock_summary(r.stdout)))

    # If hosts failed without a password, prompt and retry
    if still_failed and not dry_run:
        sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)
        for h in list(still_failed):
            r = run_sudo_script_on_host(
                h,
                fallback_script,
                sudo_password,
                ssh_kwargs=ssh_kwargs,
                timeout=_APPLY_TIMEOUT_S,
                dry_run=dry_run,
            )
            result_map[h] = r
            if r.success:
                click.echo("  [OK]   %s: %s" % (h, _gpu_clock_summary(r.stdout)))

    ok_count = sum(1 for h in host_list if result_map.get(h) and result_map[h].success)
    fail_count = sum(1 for h in host_list if result_map.get(h) and not result_map[h].success)

    for h in host_list:
        r = result_map.get(h)
        if r and not r.success:
            click.echo("  [FAIL] %s: %s" % (h, (r.stderr or r.stdout).strip()[:200]), err=True)

    click.echo()
    parts = []
    if ok_count:
        parts.append("%d applied" % ok_count)
    if fail_count:
        parts.append("%d failed" % fail_count)
    click.echo("Results: %s." % ", ".join(parts) if parts else "No hosts processed.")

    # A pre-existing unit outranks this lock at the next boot — say so.
    if unit_action == "keep" and any("UNIT: present" in (result_map.get(h).stdout if result_map.get(h) else "") for h in host_list):
        click.echo()
        click.echo("Warning: a boot-time unit is installed and still holds its own value.")
        click.echo("         Re-run with --persistent to update it, or --clear to remove it.")

    if fail_count:
        sys.exit(1)
