"""``sparkrun setup rdma-test`` — verify the high-speed fabric actually works.

Thin renderer: the api layer holds all logic; this parses flags, resolves
hosts, calls :func:`sparkrun.api.setup.rdma_test` and prints the result.

Two gates, split at the suite boundary because they gate *maturity*: the
proven perftest suite ships on every channel under ``cli.setup.rdma_test``
(a kill switch), while the young NCCL collective rides alpha under
``cli.setup.rdma_test.nccl``. See
:data:`sparkrun.core.features.FEATURE_CLI_SETUP_RDMA_TEST_NCCL`.

A gated suite is **omitted** from ``--help`` rather than listed as
unavailable, so the help describes what this install actually does. That
shaping is import-time and display-only; the refusal reads the real config
at runtime. Both go through :func:`~sparkrun.api.setup._rdma.available_suites`,
so what the help offers and what the command accepts cannot diverge.
"""

from __future__ import annotations

import sys
import textwrap

import click

from sparkrun import api
from sparkrun.api.setup import STATUS_FAIL, STATUS_OK, STATUS_SKIP, STATUS_WARN
from sparkrun.api.setup._rdma import (
    ALL_SUITES,
    SUITE_ALL,
    SUITE_DESCRIPTIONS,
    SUITE_NCCL,
    SUITE_PERFTEST,
    available_suites,
    gated_suite_message,
)

from . import setup
from .._common import _get_context, _resolve_setup_context, dry_run_option, host_options, json_option, print_json

SETUP_RDMA_TEST_FEATURE = "cli.setup.rdma_test"

_STATUS_MARK = {
    STATUS_OK: "[OK]  ",
    STATUS_WARN: "[WARN]",
    STATUS_FAIL: "[FAIL]",
    STATUS_SKIP: "[--]  ",
}

_STATUS_COLOR = {
    STATUS_OK: "green",
    STATUS_WARN: "yellow",
    STATUS_FAIL: "red",
    STATUS_SKIP: None,
}


_UNSET = object()


def _import_config():
    """Best-effort config for import-time help shaping (``None`` when unreadable).

    Memoized: the command's visibility, its help body and its ``--suite``
    metavar all need it, and ``init_sparkrun`` already runs on every shell
    completion — three config reads per TAB would land on the interactive path.
    """
    if _import_config._cached is _UNSET:
        try:
            from sparkrun.core.config import SparkrunConfig

            _import_config._cached = SparkrunConfig()
        except Exception:  # noqa: BLE001 — never let a config read break CLI import
            _import_config._cached = None
    return _import_config._cached


_import_config._cached = _UNSET


def _rdma_test_enabled_at_import() -> bool:
    """Best-effort flag resolution for help visibility (hidden when unknown)."""
    config = _import_config()
    if config is None:
        return False
    try:
        return config.is_feature_enabled(SETUP_RDMA_TEST_FEATURE)
    except Exception:  # noqa: BLE001 — never let a config read break CLI import
        return False


def _suites_at_import() -> tuple[str, ...]:
    """Suites to advertise in ``--help``.

    Display only, and deliberately so: an import-time read cannot see a
    ``--config`` override, so the *authoritative* refusal lives in
    :func:`sparkrun.api.setup.rdma_test`, which resolves against the real
    config. Both read :func:`available_suites`, so the help and the refusal
    cannot name different sets. Falls back to advertising only the ungated
    suite when the config is unreadable — claiming a gated one we cannot
    confirm is the worse error.
    """
    try:
        return available_suites(_import_config()) or (SUITE_PERFTEST,)
    except Exception:  # noqa: BLE001 — never let a config read break CLI import
        return (SUITE_PERFTEST,)


class _SuiteChoice(click.Choice):
    """Accepts every suite, but completes only the available ones.

    The full choice list is kept so a gated value reaches the api layer and
    earns its actionable "enable it with ..." error, rather than Click's
    generic "'nccl' is not one of 'perftest'" — which names no way forward.
    Completion and the metavar advertise only what will actually run.
    """

    def __init__(self, available: tuple[str, ...]) -> None:
        super().__init__(list(ALL_SUITES))
        self.available = available

    def shell_complete(self, ctx, param, incomplete):
        return [item for item in super().shell_complete(ctx, param, incomplete) if item.value in self.available]


def _suite_option_help(available: tuple[str, ...]) -> str:
    if SUITE_NCCL in available:
        return "Which tests to run ('nccl' and 'all' pull the test image and take minutes)"
    return "Which tests to run"


def _rdma_help(available: tuple[str, ...]) -> str:
    """Build the command help, omitting suites this channel cannot run.

    A gated suite is left out entirely rather than listed as unavailable: the
    help describes what this install does, and an entry the user cannot act on
    is noise in the one place they look when the fabric is already broken.
    """
    summary = "Measure RDMA bandwidth, latency and NCCL throughput across the fabric."
    if SUITE_NCCL not in available:
        summary = "Measure RDMA bandwidth and latency across the fabric."

    # A `\b` block is emitted verbatim, so the hanging indent is ours to do —
    # Click will not rewrap these lines.
    suites = "\n".join(
        textwrap.fill(
            SUITE_DESCRIPTIONS[name],
            width=72,
            initial_indent="  %-9s " % name,
            subsequent_indent=" " * 12,
        )
        for name in available
    )

    examples = ["  sparkrun setup rdma-test --cluster mylab"]
    if SUITE_ALL in available:
        examples.append("  sparkrun setup rdma-test --cluster mylab --suite all")
    examples.append("  sparkrun setup rdma-test --hosts 10.0.0.1,10.0.0.2 --json")

    return """{summary}

Verifies what `sparkrun setup cx7` configured. Host pairs are derived from the
configured CX7 subnets, so only links that physically exist are tested. Each
link is measured on its own, then every link between a host pair is driven
concurrently — on DGX Spark a single QSFP cable presents as two RDMA devices
and only shows its real throughput when both run at once.

\b
Suites:
{suites}

Underperformance is reported as a warning; only a test that could not run at
all fails the command.

\b
Examples:
{examples}
""".format(summary=summary, suites=suites, examples="\n".join(examples))


def _echo_status(status: str, text: str, indent: str = "  ") -> None:
    click.secho("%s%s %s" % (indent, _STATUS_MARK.get(status, "      "), text), fg=_STATUS_COLOR.get(status))


def _render(report) -> None:
    """Print the report. Numbers are the product; verdicts are a convenience."""
    for warning in report.warnings:
        click.secho("Note: %s" % warning, fg="yellow")
    if report.warnings:
        click.echo()

    if report.dry_run:
        # Nothing was probed, so "host-native perftest" would be a claim about
        # hosts we never looked at.
        click.echo("Tooling: %s" % (report.image if report.used_container else "host perftest, or %s" % report.image))
    elif report.used_container:
        click.echo("Tooling: %s" % report.image)
    else:
        click.echo("Tooling: host-native perftest")
    click.echo()

    if not report.pairs and report.nccl is None:
        click.echo("No tests were run.")
        return

    for pair in report.pairs:
        click.echo("%s" % pair.label)
        for link in pair.links:
            bits = []
            if link.latency is not None:
                bits.append("lat %.2f us" % link.latency.t_typical)
            if link.bandwidth is not None:
                bits.append("bw %.1f Gb/s" % link.bandwidth.avg_gbps)
            if link.expected_gbps:
                bits.append("of %.0f Gb/s" % link.expected_gbps)
            if report.dry_run:
                summary = link.link.describe()
            else:
                summary = "%s — %s" % (link.link.describe(), ", ".join(bits) if bits else "no measurement")
            _echo_status(link.status, summary, indent="    ")
            if link.detail and link.status in (STATUS_WARN, STATUS_FAIL):
                click.echo("           → %s" % link.detail)

        if pair.aggregate_gbps is not None:
            agg = "aggregate (all links concurrently) — %.1f Gb/s" % pair.aggregate_gbps
            if pair.expected_aggregate_gbps:
                agg += " of %.0f Gb/s" % pair.expected_aggregate_gbps
            click.echo("    %s %s" % ("      ", agg))
        if pair.detail:
            click.echo("           → %s" % pair.detail)
        click.echo()

    if report.nccl is not None:
        n = report.nccl
        label = "NCCL %s (%s) across %d host(s)" % (n.binary, n.msg_size, len(n.hosts))
        if n.sample is not None:
            label += " — avg bus bandwidth %.2f GB/s" % n.sample.avg_bus_bw
        _echo_status(n.status, label, indent="  ")
        if n.detail and n.status in (STATUS_WARN, STATUS_FAIL):
            click.echo("         → %s" % n.detail)
        if n.status == STATUS_FAIL and n.raw:
            click.echo()
            click.echo("  Last output from the collective:")
            for line in n.raw.strip().splitlines()[-15:]:
                click.echo("    %s" % line)
        click.echo()


def _report_to_dict(report) -> dict:
    return {
        "suite": report.suite,
        "hosts": list(report.hosts),
        "image": report.image,
        "used_container": report.used_container,
        "dry_run": report.dry_run,
        "warnings": list(report.warnings),
        "ok": report.ok_count,
        "warn": report.warn_count,
        "fail": report.fail_count,
        "pairs": [
            {
                "host_a": p.pair.host_a,
                "host_b": p.pair.host_b,
                "status": p.status,
                "detail": p.detail,
                "aggregate_gbps": p.aggregate_gbps,
                "expected_aggregate_gbps": p.expected_aggregate_gbps,
                "links": [
                    {
                        "subnet": link.link.subnet,
                        "host_a": link.link.host_a,
                        "hca_a": link.link.hca_a,
                        "ip_a": link.link.ip_a,
                        "host_b": link.link.host_b,
                        "hca_b": link.link.hca_b,
                        "ip_b": link.link.ip_b,
                        "status": link.status,
                        "detail": link.detail,
                        "expected_gbps": link.expected_gbps,
                        "latency_us_typical": link.latency.t_typical if link.latency else None,
                        "latency_us_p99": link.latency.p99 if link.latency else None,
                        "bandwidth_gbps_avg": link.bandwidth.avg_gbps if link.bandwidth else None,
                        "bandwidth_gbps_peak": link.bandwidth.peak_gbps if link.bandwidth else None,
                    }
                    for link in p.links
                ],
            }
            for p in report.pairs
        ],
        "nccl": (
            {
                "binary": report.nccl.binary,
                "msg_size": report.nccl.msg_size,
                "hosts": list(report.nccl.hosts),
                "status": report.nccl.status,
                "detail": report.nccl.detail,
                "avg_bus_bw_gbytes": report.nccl.sample.avg_bus_bw if report.nccl.sample else None,
            }
            if report.nccl is not None
            else None
        ),
    }


_SUITES_AT_IMPORT = _suites_at_import()


@setup.command("rdma-test", hidden=not _rdma_test_enabled_at_import(), help=_rdma_help(_SUITES_AT_IMPORT))
@host_options
@click.option("--user", "-u", default=None, help="SSH username (default: from config or current user)")
@click.option(
    "--suite",
    type=_SuiteChoice(_SUITES_AT_IMPORT),
    default=SUITE_PERFTEST,
    show_default=True,
    metavar="[%s]" % "|".join(_SUITES_AT_IMPORT),
    help=_suite_option_help(_SUITES_AT_IMPORT),
)
@click.option("--image", default=None, help="Test container image (default: the pinned sparkrun image)")
@click.option("-D", "--duration", default=10, show_default=True, type=int, help="Seconds per bandwidth test")
@click.option(
    "--size",
    "msg_size",
    default="16G",
    show_default=True,
    # NCCL-only, so it is omitted alongside the suite it configures rather
    # than left advertising a knob nothing on this channel reads.
    hidden=SUITE_NCCL not in _SUITES_AT_IMPORT,
    help="NCCL message size",
)
@click.option("--queue-pairs", "-q", default=4, show_default=True, type=int, help="perftest queue pairs")
@click.option("--gid-index", default=None, type=int, help="Force a RoCE GID index (normally auto)")
@click.option(
    "--link-type",
    type=click.Choice(["IB", "Ethernet", "auto"]),
    default="IB",
    show_default=True,
    help="perftest --force-link; 'auto' omits the flag",
)
@click.option("--force-container", is_flag=True, help="Use the image even when perftest is on the host")
@click.option("--keep-containers", is_flag=True, help="Leave test containers running for debugging")
@json_option(help="Emit the full results as JSON")
@dry_run_option
@click.pass_context
def setup_rdma_test(
    ctx,
    hosts,
    hosts_file,
    cluster_name,
    user,
    suite,
    image,
    duration,
    msg_size,
    queue_pairs,
    gid_index,
    link_type,
    force_container,
    keep_containers,
    output_json,
    dry_run,
):
    """Measure RDMA bandwidth, latency and (when enabled) NCCL throughput.

    The user-facing help is built by :func:`_rdma_help` and passed as ``help=``
    so it can omit suites this channel cannot run; Click prefers that over this
    docstring, which stays for developers.
    """
    sctx = _get_context(ctx)
    if not sctx.config.is_feature_enabled(SETUP_RDMA_TEST_FEATURE):
        raise click.ClickException("'setup rdma-test' is disabled. Enable it with: sparkrun setup features enable cli.setup.rdma_test")

    # Fail before the banner, so "Testing RDMA fabric across N host(s)" is
    # never printed for a run that is about to be refused. Resolved from the
    # same `available_suites` and the same config the api layer enforces
    # with — a fast path to the identical verdict, not a second opinion.
    if suite not in available_suites(sctx.config):
        raise click.ClickException(gated_suite_message(suite))

    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, sctx.config, user)

    if len(host_list) < 2:
        raise click.ClickException(
            "RDMA testing needs at least two hosts; got %d. Name a multi-host cluster with --cluster or --hosts." % len(host_list)
        )

    if not output_json:
        click.echo("Testing RDMA fabric across %d host(s): %s" % (len(host_list), ", ".join(host_list)))
        if not dry_run:
            # Set expectations honestly per suite: perftest is seconds, while
            # the collective pulls a ~1 GB image onto every host first.
            if suite == "perftest":
                click.echo("Running real RDMA transfers; this takes a few moments.")
            else:
                click.echo("Running real transfers and an NCCL collective; this takes several minutes.")
        click.echo()

    try:
        report = api.setup.rdma_test(
            sctx,
            host_list,
            ssh_kwargs,
            suite=suite,
            image=image,
            duration=duration,
            queue_pairs=queue_pairs,
            msg_size=msg_size,
            gid_index=gid_index,
            link_type=None if link_type == "auto" else link_type,
            force_container=force_container,
            keep_containers=keep_containers,
            dry_run=dry_run,
        )
    except api.setup.RdmaTestError as exc:
        raise click.ClickException(str(exc)) from exc

    if output_json:
        print_json(_report_to_dict(report))
    else:
        _render(report)
        if dry_run:
            click.echo("[dry-run] Would test %d host pair(s); no containers started and no traffic sent." % len(report.pairs))
        else:
            parts = []
            if report.ok_count:
                parts.append("%d OK" % report.ok_count)
            if report.warn_count:
                parts.append("%d warning" % report.warn_count)
            if report.fail_count:
                parts.append("%d failed" % report.fail_count)
            click.echo("Results: %s." % ", ".join(parts) if parts else "Results: nothing measured.")

    if report.has_failure:
        sys.exit(1)
