"""RDMA fabric verification — console-free implementation of ``setup rdma-test``.

Orchestrates the mechanism in :mod:`sparkrun.orchestration.rdma` into a report.
Host resolution is the caller's job (the setup-command convention): these take
an already-resolved ``host_list`` + ``ssh_kwargs``, not a cluster selector.

Three things here are policy rather than mechanism, and each is deliberate:

- **Verdicts are advisory; numbers are the product.** A link that measures
  below expectation is a ``WARN``, never a ``FAIL``. Only a test that could
  not run at all fails, because "your fabric is slower than we hoped" and
  "your fabric does not work" call for completely different responses, and
  a command that conflates them teaches people to ignore it.
- **The expectation is derived, not hardcoded.** It comes from the link's own
  sysfs-reported rate (:func:`~sparkrun.orchestration.rdma.rate_to_gbps`), so
  the check is meaningful on hardware that is not a DGX Spark. Hardcoding
  "100 Gb/s" would misreport every other NVIDIA platform.
- **The container is required only by the NCCL suite.** perftest ships
  preinstalled on DGX OS and needs no CUDA, so the common "did my cable
  work?" check costs no image pull. The image is a fallback there and a
  requirement only for the collective.
"""

from __future__ import annotations

from sparkrun.core.operation import submit as _operation_submit

import ipaddress
import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Callable, TypedDict, TypeVar

from sparkrun.api.setup._errors import RdmaTestError
from sparkrun.api.setup._rdma_plan import RdmaCoverage, plan_rdma_tests

from sparkrun.core.progress import PROGRESS
from sparkrun.orchestration.rdma import (
    DEFAULT_PERFTEST_PORT,
    BwSample,
    LatSample,
    NcclSample,
    RdmaHostFacts,
    RdmaLink,
    RdmaPair,
    build_nccl_test_cmd,
    build_perftest_cmd,
    pair_test_rounds,
    select_quick_pairs,
    parse_device_facts,
    parse_framed_runs,
    parse_nccl_busbw,
    parse_perftest_bw,
    parse_perftest_lat,
)
from sparkrun.utils.shell import quote
from sparkrun.utils.text import excerpt

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition

logger = logging.getLogger(__name__)

#: Progress reporting goes through the shared ``sparkrun.progress`` logger at
#: PROGRESS level (25), the same channel the launch pipeline uses: visible at
#: default verbosity, silenced by ``-q``, and *not* a console write — the api
#: layer still never touches stdout. Without it this command sat silent for
#: minutes through an image pull and a fan-out of real transfers, which is
#: exactly the unattributable pause `hub_degraded_message` exists to avoid
#: elsewhere.
_progress = logging.getLogger("sparkrun.progress")


def _say(msg: str, *args) -> None:
    """Emit one progress line (lazy %-formatting, per the logging convention)."""
    _progress.log(PROGRESS, msg, *args)


_T = TypeVar("_T")

#: Seconds between "still running" lines while a step is silent.
HEARTBEAT_INTERVAL_S = 30.0


def _with_heartbeat(label: str, fn: Callable[[], _T], *, budget_s: int | None = None, interval_s: float = HEARTBEAT_INTERVAL_S) -> _T:
    """Run *fn*, emitting a progress line every *interval_s* until it returns.

    The image fetch and the collective each capture their output until they
    finish, so without this they print one line and then nothing for minutes.
    A collective stuck in ``MPI_Init`` and one moving 16 GB look the same
    from outside, and the user Ctrl-Cs a run that was about to finish. The
    elapsed time, and the budget when there is one, separates them.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    future = _operation_submit(pool, copy_context().run, fn)
    start = time.monotonic()
    try:
        while True:
            try:
                result = future.result(timeout=interval_s)
            except TimeoutError:
                elapsed = int(time.monotonic() - start)
                if budget_s:
                    _say("  %s: still running (%ds elapsed, gives up at %ds)", label, elapsed, budget_s)
                else:
                    _say("  %s: still running (%ds elapsed)", label, elapsed)
                continue
            pool.shutdown(wait=True)
            return result
    except BaseException:
        # Ctrl-C reaches the ssh child through the terminal's process group;
        # waiting here would only hold the interrupt until it exits anyway.
        pool.shutdown(wait=False, cancel_futures=True)
        raise


__all__ = [
    "SUITE_PERFTEST",
    "SUITE_NCCL",
    "SUITE_ALL",
    "ALL_SUITES",
    "SUITE_DESCRIPTIONS",
    "available_suites",
    "STATUS_OK",
    "STATUS_WARN",
    "STATUS_FAIL",
    "STATUS_SKIP",
    "DEFAULT_RDMA_TEST_IMAGE",
    "DEFAULT_PARALLEL_PAIRS",
    "LinkTestResult",
    "AggregateTestResult",
    "RdmaCoverage",
    "PairTestResult",
    "NcclTestResult",
    "RdmaTestReport",
    "rdma_test",
]

SUITE_PERFTEST = "perftest"
SUITE_NCCL = "nccl"
SUITE_ALL = "all"

#: Every suite this module can run, in display order.
ALL_SUITES = (SUITE_PERFTEST, SUITE_NCCL, SUITE_ALL)

#: Suites whose maturity is still gated, and the flag gating each.
#:
#: perftest is proven and ungated. The collective is not: mpirun across
#: containers, the purpose-built nccl-tests image and a bus-bandwidth verdict
#: are all young, so they ride alpha (see
#: :data:`sparkrun.core.features.FEATURE_CLI_SETUP_RDMA_TEST_NCCL`). Note the
#: gate is *maturity*, not "needs the container" — the perftest suite falls
#: back to the image on a host without perftest and is deliberately unaffected.
_GATED_SUITES = {SUITE_NCCL: "cli.setup.rdma_test.nccl", SUITE_ALL: "cli.setup.rdma_test.nccl"}

#: One-line description per suite, shared by the CLI help and this module.
SUITE_DESCRIPTIONS = {
    SUITE_PERFTEST: "ib_write_lat + ib_write_bw on every path in both directions (default; host-native when installed)",
    SUITE_NCCL: "an nccl-tests collective through mpirun (pulls the test image onto every host; takes minutes)",
    SUITE_ALL: "both",
}


def suite_feature_flag(suite: str) -> str | None:
    """Return the flag gating *suite*, or ``None`` when it is ungated."""
    return _GATED_SUITES.get(suite)


def gated_suite_message(suite: str) -> str:
    """The refusal wording for a gated *suite*.

    One definition because two callers refuse: the CLI fails fast (before its
    "Testing RDMA fabric..." banner can claim work that will not happen) and
    this module enforces for every caller. Two spellings of the same refusal
    would drift, and the enable instruction is the part that must not.
    """
    return "the %r suite is experimental and disabled on this channel. Enable it with: sparkrun setup features enable %s" % (
        suite,
        suite_feature_flag(suite),
    )


def available_suites(config=None) -> tuple[str, ...]:
    """Return the suites this install may run, in display order.

    The single source of truth for suite availability: the CLI renders its
    help and metavar from this, and :func:`rdma_test` refuses from it. A help
    text that advertised a suite the launch would reject — or hid one it would
    accept — is read as an answer and is worse than none.

    Args:
        config: Config to resolve the gates against. ``None`` resolves against
            environment overrides and the stable channel only, which is the
            conservative reading for a caller that has no config yet.
    """
    from sparkrun.core.features import is_feature_enabled

    return tuple(s for s in ALL_SUITES if (flag := suite_feature_flag(s)) is None or is_feature_enabled(flag, config=config))


class _PerftestOptions(TypedDict, total=False):
    queue_pairs: int
    duration: int


STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"

#: Default test image, pinned rather than ``:latest``.
#:
#: The tag names the NCCL release the image was built from, which is what
#: makes "when do we rebuild?" a fact rather than a judgement call. Pinning
#: (the ``core.tooling`` philosophy) means a given sparkrun version always
#: measures with a known image, so two runs a month apart are comparable.
#: Kept in step with ``docker/rdma-test/image.env`` by
#: ``tests/test_rdma_image.py``. Overridable — the hosts that most need this
#: command are the ones that can least reach GHCR.
DEFAULT_RDMA_TEST_IMAGE = "ghcr.io/spark-arena/sparkrun-rdma-test:nccl-2.31.2-1"

#: Container name on every host. Fixed rather than random so an interrupted
#: run's leftovers are reclaimed by the next one instead of accumulating.
CONTAINER_NAME = "sparkrun-rdma-test"

#: Fraction of the link's own reported rate below which bandwidth warns.
DEFAULT_BW_WARN_RATIO = 0.75

#: Latency above this (microseconds) warns. Generous — a directly connected
#: pair measures ~1.5 us, so this only catches a link that is not really RDMA.
DEFAULT_LAT_WARN_US = 10.0

#: Cap pair workers independently of cluster size; host reservations may allow fewer.
DEFAULT_PARALLEL_PAIRS = 4

FULL_COVERAGE_HINT = "Use --full to test all candidate link paths."


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinkTestResult:
    """Latency + bandwidth in one direction; host_a sends to host_b."""

    link: RdmaLink
    latency: LatSample | None = None
    bandwidth: BwSample | None = None
    expected_gbps: float | None = None
    status: str = STATUS_SKIP
    detail: str = ""
    raw: str = ""

    @property
    def label(self) -> str:
        return self.link.describe_direction()


@dataclass(frozen=True)
class AggregateTestResult:
    """Concurrent bandwidth across a pair's paths, in one direction."""

    host_a: str
    host_b: str
    bandwidth_gbps: float | None = None
    expected_gbps: float | None = None
    status: str = STATUS_SKIP
    detail: str = ""


@dataclass(frozen=True)
class PairTestResult:
    """Both directions on every path, plus both concurrent aggregates."""

    pair: RdmaPair
    links: tuple[LinkTestResult, ...] = ()
    aggregates: tuple[AggregateTestResult, ...] = ()
    status: str = STATUS_SKIP
    detail: str = ""

    @property
    def label(self) -> str:
        return self.pair.describe()


@dataclass(frozen=True)
class NcclTestResult:
    """One nccl-tests collective across the whole cluster."""

    hosts: tuple[str, ...] = ()
    binary: str = ""
    msg_size: str = ""
    sample: NcclSample | None = None
    status: str = STATUS_SKIP
    detail: str = ""
    raw: str = ""


@dataclass
class RdmaTestReport:
    """Everything ``setup rdma-test`` measured."""

    suite: str = SUITE_PERFTEST
    hosts: tuple[str, ...] = ()
    image: str = ""
    used_container: bool = False
    dry_run: bool = False
    parallel_pairs: int = 1
    full: bool = False
    selected_pair_count: int = 0
    selected_path_count: int = 0
    host_facts: dict[str, RdmaHostFacts] = field(default_factory=dict)
    coverage: RdmaCoverage = field(default_factory=RdmaCoverage)
    pairs: tuple[PairTestResult, ...] = ()
    nccl: NcclTestResult | None = None
    warnings: list[str] = field(default_factory=list)

    def _statuses(self) -> list[str]:
        # Pair rollups and coverage are not additional measurements.
        out = [link.status for p in self.pairs for link in p.links]
        out.extend(aggregate.status for p in self.pairs for aggregate in p.aggregates)
        if self.nccl is not None:
            out.append(self.nccl.status)
        return out

    @property
    def fail_count(self) -> int:
        return sum(1 for s in self._statuses() if s == STATUS_FAIL)

    @property
    def warn_count(self) -> int:
        return sum(1 for s in self._statuses() if s == STATUS_WARN)

    @property
    def ok_count(self) -> int:
        return sum(1 for s in self._statuses() if s == STATUS_OK)

    @property
    def has_failure(self) -> bool:
        return self.fail_count > 0 or self.coverage.status == STATUS_FAIL


# ---------------------------------------------------------------------------
# Remote execution helpers
# ---------------------------------------------------------------------------


def _run_probe(hosts: list[str], ssh_kwargs: dict, *, dry_run: bool) -> dict[str, RdmaHostFacts]:
    """Run ``rdma_probe.sh`` on every host."""
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel
    from sparkrun.scripts import read_script
    from sparkrun.utils import parse_kv_output

    if dry_run:
        return {h: RdmaHostFacts(host=h, complete=False) for h in hosts}

    results = run_remote_scripts_parallel(
        list(hosts),
        read_script("rdma_probe.sh"),
        timeout=60,
        quiet=True,
        allow_local=True,
        **ssh_kwargs,
    )
    facts: dict[str, RdmaHostFacts] = {}
    for r in results:
        if not r.success:
            facts[r.host] = RdmaHostFacts(host=r.host, error=excerpt(r.stderr or "probe failed", 200))
            continue
        facts[r.host] = parse_device_facts(r.host, parse_kv_output(r.stdout))
    # Never drop a host: a missing entry and a failed one read the same to a
    # caller iterating the host list, and only one of them is true.
    for h in hosts:
        facts.setdefault(h, RdmaHostFacts(host=h, error="no result"))
    return facts


def _in_container(cmd: str, container: str | None) -> str:
    """Wrap *cmd* for ``docker exec`` when a container is in use.

    Deliberately docker, not the cluster's executor: this is the suite's own
    diagnostic container (``docker run`` with the RDMA devices and an unlimited
    memlock, removed afterwards), not a workload, and it needs docker even on a
    cluster whose workloads run under ``local`` or k8s.
    """
    if not container:
        return cmd
    return "docker exec %s bash -lc %s" % (quote(container), quote(cmd))


def _run_commands(
    host: str,
    commands: list[str],
    ssh_kwargs: dict,
    *,
    container: str | None = None,
    timeout: int = 180,
    retries: int = 0,
    pre_sleep: int = 0,
) -> dict[int, tuple[str, int]]:
    """Run *commands* concurrently on *host*; return ``{index: (output, rc)}``."""
    from sparkrun.orchestration.ssh import run_remote_script
    from sparkrun.scripts import inject_shell_vars, read_script

    wrapped = [_in_container(c, container) for c in commands]
    script = inject_shell_vars(
        read_script("rdma_perftest.sh"),
        RDMA_CMDS="\n".join(wrapped),
        RDMA_TIMEOUT=str(timeout),
        RDMA_RETRIES=str(retries),
        RDMA_PRE_SLEEP=str(pre_sleep),
    )
    # +30s so the SSH cap never fires before the script's own per-command
    # timeout does — the script's framing is what makes a timeout diagnosable.
    result = run_remote_script(
        host,
        script,
        timeout=timeout * (retries + 1) + 30,
        quiet=True,
        allow_local=True,
        session_guard=True,
        **ssh_kwargs,
    )
    if not result.success and not result.stdout:
        return {i: ((result.stderr or "").strip(), result.returncode) for i in range(len(commands))}
    runs = parse_framed_runs(result.stdout)
    if not result.success:
        # An SSH/script failure invalidates even a previously printed success
        # frame. Keep its output for diagnosis, but never discard the outer rc.
        return {
            i: (output + "\n" + (result.stderr or ""), rc or result.returncode)
            for i in range(len(commands))
            for output, rc in [runs.get(i, ("", result.returncode))]
        }
    return runs


def _run_round(
    server_host: str,
    server_cmds: list[str],
    client_host: str,
    client_cmds: list[str],
    ssh_kwargs: dict,
    *,
    container: str | None,
    timeout: int,
) -> dict[int, tuple[str, int]]:
    """Run one server/client perftest round; return the CLIENT's framed runs.

    Both sides are dispatched at the same moment over separate SSH
    connections, so the client can arrive before the server is listening.
    perftest has no "wait for peer" mode and ``-R`` (rdma_cm) opens no TCP
    listener that :func:`~sparkrun.orchestration.health.wait_for_port` could
    observe — so the rendezvous is the client's retry, plus a short head start
    for the server.

    Measurements come from the client. Both processes must exit successfully;
    a printed table alone does not certify that the transfer completed.
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        server_fut = _operation_submit(
            pool,
            _run_commands,
            server_host,
            server_cmds,
            ssh_kwargs,
            container=container,
            timeout=timeout,
            retries=0,
        )
        client_fut = _operation_submit(
            pool,
            _run_commands,
            client_host,
            client_cmds,
            ssh_kwargs,
            container=container,
            timeout=timeout,
            retries=4,
            pre_sleep=2,
        )
        try:
            client_runs = client_fut.result()
        except Exception as exc:  # noqa: BLE001 — retain a verdict for every planned transfer
            client_runs = {i: ("client execution failed: %s" % exc, -1) for i in range(len(client_cmds))}
        try:
            server_runs = server_fut.result()
        except Exception as exc:  # noqa: BLE001 — a server error must invalidate a printed client table
            server_runs = {i: ("server execution failed: %s" % exc, -1) for i in range(len(server_cmds))}
    for i in range(len(client_cmds)):
        output, rc = client_runs.get(i, ("", -1))
        server_output, server_rc = server_runs.get(i, ("no server result", -1))
        if server_rc != 0:
            client_runs[i] = (output + "\nserver failed (rc=%d): %s" % (server_rc, server_output), rc or server_rc)
    return client_runs


# ---------------------------------------------------------------------------
# Container lifecycle
# ---------------------------------------------------------------------------

# The device test is `-c` (character device), NOT `-e` (exists). /dev/infiniband
# holds two *directories* alongside the real nodes — `by-ibdev` and `by-path`,
# each a tree of symlinks — and passing either to --device fails the whole
# `docker run` with "not a device node", taking every real device with it.
# Enumerating rather than passing the directory is deliberate too: Docker's
# handling of a directory as --device is not portable across runtimes.
_START_CONTAINER = """set -uo pipefail
docker rm -f {name} >/dev/null 2>&1 || true
IB_DEV_DIR="${{SPARKRUN_IB_DEV_DIR:-/dev/infiniband}}"
DEV_OPTS=""
for d in "$IB_DEV_DIR"/*; do
    [ -c "$d" ] && DEV_OPTS="$DEV_OPTS --device=$d"
done
if [ -z "$DEV_OPTS" ]; then
    echo "no character devices under $IB_DEV_DIR to pass into the container" >&2
fi
# shellcheck disable=SC2086
docker run -d --rm --name {name} \\
    --network host --ipc host \\
    $DEV_OPTS \\
    --cap-add IPC_LOCK --ulimit memlock=-1:-1 \\
    {gpu_opts} {mounts} \\
    {image} sleep infinity >/dev/null
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "docker run failed (rc=$rc)" >&2
    exit "$rc"
fi
echo "RDMA_CONTAINER_STARTED=1"
"""

_STOP_CONTAINER = """set -uo pipefail
docker rm -f {name} >/dev/null 2>&1 || true
echo "RDMA_CONTAINER_STOPPED=1"
"""


def _start_containers(
    hosts: list[str],
    image: str,
    ssh_kwargs: dict,
    *,
    need_gpu: bool,
    mount_ssh: bool,
    dry_run: bool,
) -> tuple[list[str], dict[str, str]]:
    """Start the test container on every host.

    Returns ``(failed_hosts, reason_by_host)``. The reasons are returned
    rather than logged at debug: docker's own message names the actual
    problem ("not a device node", "could not select device driver"), and
    "check that docker is usable" is a guess by comparison — one that sent a
    real failure to a `setup check` that would have reported everything fine.
    """
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    if dry_run:
        return [], {}

    script = _START_CONTAINER.format(
        name=CONTAINER_NAME,
        image=quote(image),
        gpu_opts="--gpus all" if need_gpu else "",
        # Double-quoted so $HOME expands on the target, which is the whole
        # point — the key lives in the invoking user's home, not ours.
        mounts='-v "$HOME/.ssh":/tmp/.ssh:ro' if mount_ssh else "",
    )
    results = run_remote_scripts_parallel(list(hosts), script, timeout=180, quiet=True, allow_local=True, **ssh_kwargs)

    failed: list[str] = []
    reasons: dict[str, str] = {}
    for r in results:
        if r.success:
            continue
        failed.append(r.host)
        # docker writes the useful line to stderr; keep the last non-empty one.
        detail = ""
        for line in reversed((r.stderr or "").splitlines()):
            if line.strip() and "docker run failed" not in line:
                detail = line.strip()
                break
        reasons[r.host] = detail or "rc=%d" % r.returncode
        logger.debug("container start failed on %s: %s", r.host, reasons[r.host])
    return failed, reasons


def _stop_containers(hosts: list[str], ssh_kwargs: dict, *, dry_run: bool) -> None:
    """Best-effort teardown. Never raises — it runs from a ``finally``."""
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    if dry_run:
        return
    try:
        run_remote_scripts_parallel(
            list(hosts),
            _STOP_CONTAINER.format(name=CONTAINER_NAME),
            timeout=60,
            quiet=True,
            allow_local=True,
            **ssh_kwargs,
        )
    except Exception:  # noqa: BLE001
        logger.warning("RDMA test container teardown failed; check for leftover %s", CONTAINER_NAME)


# ---------------------------------------------------------------------------
# perftest suite
# ---------------------------------------------------------------------------


def _port_share(host: str, hca: str, links, facts: dict[str, RdmaHostFacts]) -> int:
    """How many links under test drive the same physical port on *host*.

    Several RDMA devices can be functions of one adapter sharing one wire
    (DGX Spark exposes two such functions per port). Those devices cannot
    each reach the port's full rate at once, so a per-device expectation of
    the whole rate warns on hardware that is working correctly.
    """
    f = facts.get(host)
    if f is None:
        return 1
    dev = f.device(hca)
    if dev is None:
        return 1
    key = dev.port_key
    count = 0
    for link in links:
        other = link.hca_a if link.host_a == host else (link.hca_b if link.host_b == host else "")
        if not other:
            continue
        d = f.device(other)
        if d is not None and d.port_key == key:
            count += 1
    return max(count, 1)


def _expected_gbps(link: RdmaLink, facts: dict[str, RdmaHostFacts], links=()) -> float | None:
    """Expected per-device bandwidth: the port's rate, divided by its sharers.

    The slower end wins, and the figure is the device's *fair share* of its
    physical port rather than the port's whole rate — on DGX Spark two PCIe
    functions share one 200 Gb/s port, so a full-rate expectation reports a
    healthy ~112 Gb/s link as underperforming.
    """
    rates = []
    for host, hca in ((link.host_a, link.hca_a), (link.host_b, link.hca_b)):
        f = facts.get(host)
        dev = f.device(hca) if f else None
        if dev and dev.rate_gbps:
            rates.append(dev.rate_gbps / _port_share(host, hca, links or (link,), facts))
    return min(rates) if rates else None


def _run_pairs(
    pairs: list[RdmaPair],
    run_pair: Callable[[RdmaPair], PairTestResult],
    parallel_pairs: int,
) -> tuple[PairTestResult, ...]:
    """Run disjoint pairs concurrently, reserving both hosts for a whole pair.

    Reservations cover both directions and aggregates. Results retain the
    original candidate order, regardless of completion order. On an exception,
    the pool drains active work before the caller can tear down containers.
    """
    total = len(pairs)
    finished = 0

    def _done(pair: RdmaPair, result: PairTestResult) -> PairTestResult:
        nonlocal finished
        finished += 1
        if total > 1:
            _say("[%d/%d] %s <-> %s: %s", finished, total, pair.host_a, pair.host_b, result.status.upper())
        return result

    if parallel_pairs == 1 or len(pairs) < 2:
        return tuple(_done(pair, run_pair(pair)) for pair in pairs)
    batches = pair_test_rounds(pairs)
    busy: set[str] = set()
    results: dict[tuple[str, str], PairTestResult] = {}
    with ThreadPoolExecutor(max_workers=parallel_pairs) as pool:
        active = {}
        while batches or active:
            while batches and len(active) < parallel_pairs:
                pending = batches[0]
                for pair in list(pending):
                    if len(active) == parallel_pairs:
                        break
                    if busy.intersection(pair.key):
                        continue
                    busy.update(pair.key)
                    active[_operation_submit(pool, copy_context().run, run_pair, pair)] = pair
                    pending.remove(pair)
                if pending:
                    # Preserve matching priority: later pairs must not steal
                    # hosts from this round and strand its remaining pairs.
                    break
                batches.pop(0)
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                pair = active.pop(future)
                busy.difference_update(pair.key)
                results[pair.key] = _done(pair, future.result())
    return tuple(results[pair.key] for pair in pairs)


def _run_pair(
    pair: RdmaPair,
    ssh_kwargs: dict,
    facts: dict[str, RdmaHostFacts],
    *,
    container: str | None,
    duration: int,
    queue_pairs: int,
    gid_index: int | None,
    link_type: str | None,
    bw_warn_ratio: float,
    lat_warn_us: float,
) -> PairTestResult:
    """Measure every path and the concurrent aggregate in both directions."""
    link_results: list[LinkTestResult] = []
    aggregates: list[AggregateTestResult] = []
    timeout = max(duration * 4, 60)
    _say("Testing %s (both directions)", pair.describe())
    reverse = RdmaPair(pair.host_b, pair.host_a, tuple(link.reversed() for link in pair.links))
    for directed_pair in (pair, reverse):
        results, aggregate = _run_direction(
            directed_pair,
            ssh_kwargs,
            facts,
            container=container,
            duration=duration,
            queue_pairs=queue_pairs,
            gid_index=gid_index,
            link_type=link_type,
            bw_warn_ratio=bw_warn_ratio,
            lat_warn_us=lat_warn_us,
            timeout=timeout,
        )
        link_results.extend(results)
        if aggregate is not None:
            aggregates.append(aggregate)
    statuses = [r.status for r in link_results] + [a.status for a in aggregates]
    return PairTestResult(pair=pair, links=tuple(link_results), aggregates=tuple(aggregates), status=_rollup(statuses))


def _rollup(statuses: list[str]) -> str:
    for status in (STATUS_FAIL, STATUS_WARN, STATUS_OK):
        if status in statuses:
            return status
    return STATUS_SKIP


def _run_direction(
    pair: RdmaPair,
    ssh_kwargs: dict,
    facts: dict[str, RdmaHostFacts],
    *,
    container: str | None,
    duration: int,
    queue_pairs: int,
    gid_index: int | None,
    link_type: str | None,
    bw_warn_ratio: float,
    lat_warn_us: float,
    timeout: int,
) -> tuple[list[LinkTestResult], AggregateTestResult | None]:
    link_results: list[LinkTestResult] = []
    for idx, link in enumerate(pair.links):
        port = DEFAULT_PERFTEST_PORT + idx
        expected = _expected_gbps(link, facts, pair.links)

        if not link.usable:
            link_results.append(
                LinkTestResult(
                    link=link,
                    expected_gbps=expected,
                    status=STATUS_SKIP,
                    detail="link is missing an RDMA device or address on one end",
                )
            )
            continue

        lat_sample: LatSample | None = None
        bw_sample: BwSample | None = None
        raw_parts: list[str] = []
        failures: list[str] = []

        rounds: tuple[tuple[str, _PerftestOptions], ...] = (
            ("ib_write_lat", {}),
            ("ib_write_bw", {"queue_pairs": queue_pairs, "duration": duration}),
        )
        for tool, kwargs in rounds:
            _say("  %s: %s", tool, link.describe_direction())
            server_cmd = build_perftest_cmd(
                tool, link.hca_b, source_ip=link.ip_b, port=port, gid_index=gid_index, link_type=link_type, **kwargs
            )
            client_cmd = build_perftest_cmd(
                tool,
                link.hca_a,
                peer_ip=link.ip_b,
                source_ip=link.ip_a,
                port=port,
                gid_index=gid_index,
                link_type=link_type,
                **kwargs,
            )
            runs = _run_round(
                link.host_b,
                [server_cmd],
                link.host_a,
                [client_cmd],
                ssh_kwargs,
                container=container,
                timeout=timeout,
            )
            output, rc = runs.get(0, ("", -1))
            raw_parts.append("$ %s\n%s" % (client_cmd, output))
            if rc != 0:
                failures.append("%s failed (rc=%d)" % (tool, rc))
            if tool == "ib_write_lat":
                lat_sample = parse_perftest_lat(output)
                if lat_sample is None:
                    failures.append("latency test produced no result (rc=%d)" % rc)
            else:
                bw_sample = parse_perftest_bw(output)
                if bw_sample is None:
                    failures.append("bandwidth test produced no result (rc=%d)" % rc)

        status, detail = _grade_link(lat_sample, bw_sample, expected, failures, bw_warn_ratio, lat_warn_us)
        link_results.append(
            LinkTestResult(
                link=link,
                latency=lat_sample,
                bandwidth=bw_sample,
                expected_gbps=expected,
                status=status,
                detail=detail,
                raw="\n\n".join(raw_parts),
            )
        )

    aggregate = _run_aggregate(
        pair,
        ssh_kwargs,
        facts,
        container=container,
        duration=duration,
        queue_pairs=queue_pairs,
        gid_index=gid_index,
        link_type=link_type,
        timeout=timeout,
        bw_warn_ratio=bw_warn_ratio,
    )
    return link_results, aggregate


def _run_aggregate(
    pair: RdmaPair,
    ssh_kwargs: dict,
    facts: dict[str, RdmaHostFacts],
    *,
    container: str | None,
    duration: int,
    queue_pairs: int,
    gid_index: int | None,
    link_type: str | None,
    timeout: int,
    bw_warn_ratio: float = DEFAULT_BW_WARN_RATIO,
) -> AggregateTestResult | None:
    """Drive every path between the hosts concurrently, in one direction.

    Skipped for a single-link pair, where it would repeat the per-link number
    exactly and double the runtime for nothing.
    """
    usable = [link for link in pair.links if link.usable]
    if len(usable) < 2:
        return None

    server_cmds, client_cmds = [], []
    for idx, link in enumerate(usable):
        port = DEFAULT_PERFTEST_PORT + 100 + idx
        server_cmds.append(
            build_perftest_cmd(
                "ib_write_bw",
                link.hca_b,
                source_ip=link.ip_b,
                port=port,
                queue_pairs=queue_pairs,
                duration=duration,
                gid_index=gid_index,
                link_type=link_type,
            )
        )
        client_cmds.append(
            build_perftest_cmd(
                "ib_write_bw",
                link.hca_a,
                peer_ip=link.ip_b,
                source_ip=link.ip_a,
                port=port,
                queue_pairs=queue_pairs,
                duration=duration,
                gid_index=gid_index,
                link_type=link_type,
            )
        )

    _say("  ib_write_bw: %s -> %s, %d device(s) concurrently", pair.host_a, pair.host_b, len(usable))
    runs = _run_round(pair.host_b, server_cmds, pair.host_a, client_cmds, ssh_kwargs, container=container, timeout=timeout)

    total = 0.0
    failures: list[str] = []
    for i, link in enumerate(usable):
        output, rc = runs.get(i, ("", -1))
        sample = parse_perftest_bw(output)
        if rc != 0 or sample is None:
            failures.append("%s: %s (rc=%d)" % (link.describe_direction(), "failed" if rc else "no result", rc))
        else:
            total += sample.avg_gbps

    # Both endpoint capacities constrain the aggregate. Devices on one
    # physical port share its rate; unknown rates cannot establish a ceiling.
    capacities: list[float] = []
    for host, hcas in ((pair.host_a, [link.hca_a for link in usable]), (pair.host_b, [link.hca_b for link in usable])):
        by_port: dict[tuple[str, str], float] = {}
        f = facts.get(host)
        for hca in hcas:
            dev = f.device(hca) if f else None
            if dev is None or not dev.rate_gbps:
                break
            by_port[dev.port_key] = max(by_port.get(dev.port_key, 0.0), dev.rate_gbps)
        else:
            capacities.append(sum(by_port.values()))
    expected = min(capacities) if len(capacities) == 2 else None
    if failures:
        # A partial total must not masquerade as the pair's bandwidth.
        return AggregateTestResult(pair.host_a, pair.host_b, expected_gbps=expected, status=STATUS_FAIL, detail="; ".join(failures))
    status, detail = STATUS_OK, ""
    if expected and total < expected * bw_warn_ratio:
        status = STATUS_WARN
        detail = "aggregate bandwidth %.1f Gb/s is below %.0f%% of %.1f Gb/s" % (total, bw_warn_ratio * 100, expected)
    return AggregateTestResult(pair.host_a, pair.host_b, total, expected, status, detail)


def _grade_link(
    lat: LatSample | None,
    bw: BwSample | None,
    expected: float | None,
    failures: list[str],
    bw_warn_ratio: float,
    lat_warn_us: float,
) -> tuple[str, str]:
    """Turn samples into a verdict.

    A failed command or missing measurement is a FAIL: the run did not finish
    on that link. Underperformance is a WARN — it is a real finding, but it
    does not stop a workload, and reporting it at the same severity as a dead
    link makes the severe case invisible.
    """
    if failures:
        return STATUS_FAIL, "; ".join(failures)

    notes = []
    status = STATUS_OK

    if bw is not None and expected:
        if bw.avg_gbps < expected * bw_warn_ratio:
            status = STATUS_WARN
            notes.append(
                "bandwidth %.1f Gb/s is below %.0f%% of the link's %.0f Gb/s rate"
                % (
                    bw.avg_gbps,
                    bw_warn_ratio * 100,
                    expected,
                )
            )
    if lat is not None and lat.t_typical > lat_warn_us:
        status = STATUS_WARN
        notes.append("latency %.2f us exceeds %.1f us" % (lat.t_typical, lat_warn_us))

    return status, "; ".join(notes)


# ---------------------------------------------------------------------------
# NCCL suite
# ---------------------------------------------------------------------------


_WRITE_RSH = """set -uo pipefail
cat > /tmp/sparkrun-rsh-wrapper.sh <<'SPARKRUN_RSH_EOF'
{wrapper}SPARKRUN_RSH_EOF
docker cp /tmp/sparkrun-rsh-wrapper.sh {name}:{path} || exit 1
docker exec {name} chmod +x {path} || exit 1
rm -f /tmp/sparkrun-rsh-wrapper.sh
echo "RDMA_RSH_INSTALLED=1"
"""


def _clip(text: str, head: int = 3000, tail: int = 2000) -> str:
    """Keep both ends of *text*, eliding the middle.

    Tail-only truncation loses the diagnosis for mpirun, which prints the
    actual cause first ("MCA parameter plm_rsh_agent ... could not be found")
    and a generic five-cause block last — so the useful half was exactly what
    got cut, leaving a report that named everything except what happened.
    A successful run puts its results table at the end, so neither end can be
    dropped unconditionally.
    """
    if len(text) <= head + tail:
        return text
    omitted = len(text) - head - tail
    return "%s\n... [%d bytes omitted] ...\n%s" % (text[:head], omitted, text[-tail:])


def _nccl_env(host: str, pairs: list[RdmaPair], facts: dict[str, RdmaHostFacts], detections: dict) -> dict[str, str]:
    """NCCL variables pinning the collective to the fabric under test.

    Built from the links this command already derived rather than by a second
    detection sweep, so the collective provably runs over the same devices the
    perftest suite just measured.
    """
    hcas: list[str] = []
    for pair in pairs:
        for link in pair.links:
            if link.host_a == host and link.hca_a not in hcas:
                hcas.append(link.hca_a)
            if link.host_b == host and link.hca_b not in hcas:
                hcas.append(link.hca_b)

    # HWLOC_COMPONENTS=-gl: hwloc's GL component probes local X displays, and
    # with --network host a desktop session's X server answers, so every rank
    # on such a host printed "Authorization required, but no authorization
    # protocol specified" into the report.
    env = {"NCCL_IB_DISABLE": "0", "NCCL_DEBUG": "WARN", "HWLOC_COMPONENTS": "-gl"}
    if hcas:
        env["NCCL_IB_HCA"] = ",".join(hcas)
    det = detections.get(host)
    if det is not None and getattr(det, "mgmt_iface", ""):
        env["NCCL_SOCKET_IFNAME"] = det.mgmt_iface
    return env


def _nccl_failure_detail(output: str, rc: int, timeout: int) -> str:
    """Name the failure when the output says what it was, else the exit code."""
    if rc == 124:
        return "collective timed out after %ds with no result (rc=124)" % timeout
    if "out of memory" in output:
        return "collective ran out of GPU memory (rc=%d); free memory on the hosts (sparkrun setup clear-cache) or lower --size" % rc
    return "collective %s (rc=%d)" % ("failed" if rc else "produced no bus-bandwidth figure", rc)


def _mpi_tcp_if_include(hosts: list[str], detections: dict) -> str | None:
    """The network Open MPI's own TCP traffic should be pinned to, if one is known.

    Open MPI advertises every local address for its out-of-band and TCP BTL
    traffic. A host running docker networks (``br-*`` bridges on 172.x) offers
    addresses no peer can route to, and ``MPI_Init`` blocks on them until the
    collective's timeout fires, with no output at all.

    The management subnet is preferred over an interface name because
    ``*_if_include`` refuses names that do not exist on a host. Each host's
    subnet is the most specific routed subnet containing its management IP;
    they must all agree. Otherwise a management interface name shared by
    every host is used. Otherwise ``None`` (unpinned, the previous behavior),
    since a list naming a subnet a host lacks makes Open MPI warn on that host.
    """
    cidrs: set[str] = set()
    names: set[str] = set()
    for host in hosts:
        det = detections.get(host)
        names.add(getattr(det, "mgmt_iface", "") or "")
        cidr = ""
        try:
            ip = ipaddress.ip_address(getattr(det, "mgmt_ip", "") or "")
        except ValueError:
            ip = None
        if ip is not None:
            best = None
            for raw in getattr(det, "used_subnets", ()) or ():
                try:
                    net = ipaddress.ip_network(raw, strict=False)
                except ValueError:
                    continue
                if ip in net and (best is None or net.prefixlen > best.prefixlen):
                    best = net
            cidr = str(best) if best is not None else ""
        cidrs.add(cidr)
    if len(cidrs) == 1 and "" not in cidrs:
        return cidrs.pop()
    if len(names) == 1 and "" not in names:
        return names.pop()
    logger.debug("not pinning mpirun TCP traffic: management subnets %s, interfaces %s disagree", cidrs, names)
    return None


def _run_nccl(
    hosts: list[str],
    pairs: list[RdmaPair],
    facts: dict[str, RdmaHostFacts],
    detections: dict,
    ssh_kwargs: dict,
    *,
    msg_size: str,
    binary: str,
    timeout: int,
) -> NcclTestResult:
    """Run one nccl-tests collective across every host."""
    from sparkrun.orchestration.mpi import DEFAULT_RSH_WRAPPER_PATH, build_rsh_wrapper
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    if len(hosts) < 2:
        return NcclTestResult(
            hosts=tuple(hosts),
            binary=binary,
            msg_size=msg_size,
            status=STATUS_SKIP,
            detail="a collective needs at least two hosts",
        )

    head = hosts[0]
    # The container runs as root while the cluster is reached as an ordinary
    # login, so the agent must name the user explicitly or its `ssh <host>`
    # goes out as root@ and is refused — which mpirun reports as "lack of
    # authority to execute on one or more specified nodes".
    wrapper = build_rsh_wrapper({h: CONTAINER_NAME for h in hosts}, ssh_user=ssh_kwargs.get("ssh_user"))
    install = _WRITE_RSH.format(
        wrapper=wrapper,
        name=quote(CONTAINER_NAME),
        path=DEFAULT_RSH_WRAPPER_PATH,
    )
    # EVERY host, not just the head. `plm_rsh_agent` is propagated to each
    # node's orted, which resolves the path locally — a peer without the file
    # aborts with "MCA parameter plm_rsh_agent was set to a path that could
    # not be found", and mpirun then reports the generic
    # could-not-start-daemons block that names five unrelated causes.
    installs = run_remote_scripts_parallel(list(hosts), install, timeout=60, quiet=True, allow_local=True, **ssh_kwargs)
    install_failed = [r.host for r in installs if not r.success]
    if install_failed:
        return NcclTestResult(
            hosts=tuple(hosts),
            binary=binary,
            msg_size=msg_size,
            status=STATUS_FAIL,
            detail="could not install the mpirun rsh agent on: %s" % ", ".join(install_failed),
            raw="\n".join((r.stderr or "").strip() for r in installs if not r.success)[:2000],
        )

    # The collective allocates its buffers in unified memory, where the page
    # cache is not reclaimed for cudaMalloc: a host whose cache had filled
    # free memory failed with "Test CUDA failure ... out of memory" even at a
    # 64M message. The launch path drops the cache for the same reason.
    from sparkrun.orchestration.primitives import try_clear_page_cache

    try_clear_page_cache(list(hosts), ssh_kwargs=ssh_kwargs)

    tcp_if_include = _mpi_tcp_if_include(hosts, detections)
    _say(
        "Running NCCL %s (%s) across %d host(s) via mpirun%s...",
        binary,
        msg_size,
        len(hosts),
        " (MPI traffic on %s)" % tcp_if_include if tcp_if_include else "",
    )
    env = _nccl_env(head, pairs, facts, detections)
    cmd = build_nccl_test_cmd(
        list(hosts),
        rsh_agent=DEFAULT_RSH_WRAPPER_PATH,
        binary=binary,
        msg_size=msg_size,
        env=env,
        tcp_if_include=tcp_if_include,
    )
    exported = " ".join("%s=%s" % (k, quote(v)) for k, v in sorted(env.items()))
    full = "export %s; %s" % (exported, cmd) if exported else cmd

    runs = _with_heartbeat(
        "NCCL %s" % binary,
        lambda: _run_commands(
            head,
            [full],
            ssh_kwargs,
            container=CONTAINER_NAME,
            timeout=timeout,
            retries=0,
        ),
        budget_s=timeout,
    )
    output, rc = runs.get(0, ("", -1))
    sample = parse_nccl_busbw(output)
    if sample is None or rc != 0:
        return NcclTestResult(
            hosts=tuple(hosts),
            binary=binary,
            msg_size=msg_size,
            status=STATUS_FAIL,
            detail=_nccl_failure_detail(output, rc, timeout),
            raw=_clip(output),
        )
    return NcclTestResult(
        hosts=tuple(hosts),
        binary=binary,
        msg_size=msg_size,
        sample=sample,
        status=STATUS_OK,
        raw=_clip(output),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def rdma_test(
    sctx,
    host_list: list[str],
    ssh_kwargs: dict,
    *,
    suite: str = SUITE_PERFTEST,
    cluster: ClusterDefinition | None = None,
    parallel_pairs: int = DEFAULT_PARALLEL_PAIRS,
    full: bool = False,
    image: str | None = None,
    duration: int = 10,
    queue_pairs: int = 4,
    msg_size: str = "16G",
    nccl_binary: str = "all_gather_perf",
    gid_index: int | None = None,
    link_type: str | None = "IB",
    force_container: bool = False,
    keep_containers: bool = False,
    offline: bool | None = None,
    dry_run: bool = False,
) -> RdmaTestReport:
    """Verify RDMA bandwidth, latency and (optionally) a NCCL collective.

    Args:
        sctx: Sparkrun context, for config-resolved defaults.
        host_list: Already-resolved hosts.
        ssh_kwargs: Already-built SSH parameters.
        suite: ``perftest`` (default), ``nccl`` or ``all``. perftest is the
            default because it answers the common question — is the fabric
            carrying traffic at line rate — host-native when available, with
            no image pull. Every selected path is tested in both directions.
            The collective needs the container on every host.
        cluster: Selected cluster definition, for topology and fabric interface
            constraints. Without it, coverage is limited to observed subnets.
        parallel_pairs: Maximum simultaneous host-disjoint pairs (default 4),
            bounded by the available hosts. Set to 1 for serial measurements.
        full: Test all candidate paths instead of the default spanning chain
            on each subnet. Both modes validate the complete inventory and keep
            both directions and aggregates. NCCL always uses all selected hosts.
        image: Test image override.
        duration: Seconds per bandwidth test.
        queue_pairs: perftest ``-q``.
        msg_size: nccl-tests message size.
        nccl_binary: nccl-tests binary to run.
        gid_index: perftest ``-x`` override.
        link_type: perftest ``--force-link``; ``None`` to omit.
        force_container: Use the image even when perftest is on the host.
        keep_containers: Leave containers running for debugging.
        offline: ``True`` / ``False`` for ``--offline`` / ``--online``; ``None``
            follows the cluster's ``offline:`` setting. Offline, the test image
            is only copied from where it already exists, never pulled.
        dry_run: Resolve and plan without running anything.

    Returns:
        The report. Coverage failures prevent transfers; failed or incomplete
        transfers fail the report. Underperformance produces warnings.

    Raises:
        RdmaTestError: Invalid arguments, or required containers could not be
            prepared. Discovery and coverage failures are returned in the report.
    """
    from sparkrun.core.offline import resolve_offline
    from sparkrun.orchestration.networking import detect_cx7_for_hosts

    if suite not in ALL_SUITES:
        raise RdmaTestError("unknown suite %r (expected %s)" % (suite, ", ".join(ALL_SUITES)))

    config = getattr(sctx, "config", None)
    offline_mode = resolve_offline(offline, cluster)

    # Enforced here rather than in the CLI so every caller passes the same
    # gate (the sidecar reaches this function directly), and *before* the
    # dry-run branches below — a dry run must not plan a suite the real run
    # would refuse.
    if suite not in available_suites(config):
        raise RdmaTestError(gated_suite_message(suite))

    if len(host_list) < 2 or len(set(host_list)) != len(host_list):
        raise RdmaTestError("RDMA testing requires at least two distinct hosts without duplicates")
    if parallel_pairs < 1:
        raise RdmaTestError("parallel_pairs must be positive")
    if duration < 1 or queue_pairs < 1:
        raise RdmaTestError("duration and queue_pairs must be positive")

    settings = {}
    if config is not None and hasattr(config, "rdma_test_settings"):
        settings = config.rdma_test_settings() or {}

    resolved_image = image or settings.get("image") or DEFAULT_RDMA_TEST_IMAGE
    bw_ratio = float(settings.get("bw_warn_ratio", DEFAULT_BW_WARN_RATIO))
    lat_warn = float(settings.get("lat_warn_us", DEFAULT_LAT_WARN_US))

    pair_limit = min(parallel_pairs, len(host_list) // 2)
    report = RdmaTestReport(
        suite=suite, hosts=tuple(host_list), image=resolved_image, dry_run=dry_run, parallel_pairs=pair_limit, full=full
    )

    if not dry_run:
        _say("Probing RDMA devices on %d host(s)...", len(host_list))
    facts = _run_probe(host_list, ssh_kwargs, dry_run=dry_run)
    report.host_facts = facts

    detections = detect_cx7_for_hosts(host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run)
    all_pairs, report.coverage = plan_rdma_tests(host_list, detections, facts, cluster=cluster, dry_run=dry_run)
    if report.coverage.status == STATUS_FAIL:
        return report
    wants_perftest = suite in (SUITE_PERFTEST, SUITE_ALL)
    wants_nccl = suite in (SUITE_NCCL, SUITE_ALL)
    pairs = select_quick_pairs(all_pairs) if wants_perftest and not full else all_pairs
    report.selected_pair_count = len(pairs)
    report.selected_path_count = sum(len(pair.links) for pair in pairs)
    sampled = report.selected_path_count < report.coverage.path_count
    if pairs and wants_perftest and not dry_run:
        _say(
            "Planned %d of %d candidate path(s) across %d of %d host pair(s); testing both directions%s",
            report.selected_path_count,
            report.coverage.path_count,
            report.selected_pair_count,
            report.coverage.pair_count,
            " (quick sample; skipped pairs unverified)" if sampled else "",
        )
        if sampled:
            _say(FULL_COVERAGE_HINT)

    if dry_run:
        # A dry run does not probe, so it must not claim anything about what
        # is installed. `used_container` reports only what the *suite*
        # requires; whether perftest turns out to be host-native is unknown
        # here, and asserting "perftest not found" from an empty probe is a
        # statement about the hosts we never looked at.
        report.used_container = wants_nccl or force_container
        report.pairs = tuple(
            PairTestResult(
                pair=p,
                links=tuple(
                    LinkTestResult(link=link, status=STATUS_SKIP, detail="dry run")
                    for link in (*p.links, *(candidate.reversed() for candidate in p.links))
                ),
                aggregates=tuple(AggregateTestResult(a, b, detail="dry run") for a, b in (p.key, p.key[::-1])) if len(p.links) > 1 else (),
                status=STATUS_SKIP,
            )
            for p in pairs
            if wants_perftest
        )
        if wants_nccl:
            report.nccl = NcclTestResult(
                hosts=tuple(host_list), binary=nccl_binary, msg_size=msg_size, status=STATUS_SKIP, detail="dry run"
            )
        return report

    host_native = all(facts[h].has_perftest for h in host_list) and not force_container
    # The collective lives only in the image; perftest may not need it at all.
    need_container = wants_nccl or not host_native
    report.used_container = need_container

    if not host_native and wants_perftest:
        missing = [h for h in host_list if not facts[h].has_perftest]
        if missing:
            report.warnings.append("perftest not found on %s; running it from %s" % (", ".join(missing), resolved_image))

    container: str | None = None
    try:
        if need_container:
            _prepare_container(
                host_list,
                resolved_image,
                ssh_kwargs,
                need_gpu=wants_nccl,
                mount_ssh=wants_nccl,
                config=config,
                cluster=cluster,
                offline=offline_mode.offline,
            )
            container = CONTAINER_NAME

        if wants_perftest:
            _say("Testing up to %d host-disjoint pair(s) concurrently", pair_limit)
            report.pairs = _run_pairs(
                pairs,
                partial(
                    _run_pair,
                    ssh_kwargs=ssh_kwargs,
                    facts=facts,
                    container=container,
                    duration=duration,
                    queue_pairs=queue_pairs,
                    gid_index=gid_index,
                    link_type=link_type,
                    bw_warn_ratio=bw_ratio,
                    lat_warn_us=lat_warn,
                ),
                pair_limit,
            )

        if wants_nccl:
            report.nccl = _run_nccl(
                list(host_list),
                all_pairs,
                facts,
                detections,
                ssh_kwargs,
                msg_size=msg_size,
                binary=nccl_binary,
                timeout=max(duration * 30, 600),
            )
    finally:
        if container and not keep_containers:
            _stop_containers(list(host_list), ssh_kwargs, dry_run=False)
        elif container:
            report.warnings.append("containers left running as '%s'; remove with: docker rm -f %s" % (CONTAINER_NAME, CONTAINER_NAME))

    return report


def _operation_config(config, cluster: ClusterDefinition | None, ssh_kwargs: dict):
    """The cluster-scoped config image distribution runs under, matching *ssh_kwargs*.

    Distribution builds its SSH settings from a config rather than taking
    them, so the config must name the user the probes and tests use (an
    explicit ``--user`` outranks the cluster's ``user:``). Otherwise the image
    would be fetched as one user and the containers started as another.
    """
    from copy import copy

    from sparkrun.orchestration.primitives import build_ssh_kwargs

    scoped = config.for_cluster(cluster)
    user = ssh_kwargs.get("ssh_user")
    if user and build_ssh_kwargs(scoped).get("ssh_user") != user:
        scoped = copy(scoped)
        scoped.ssh_user = user
    return scoped


def _prepare_container(
    host_list: list[str],
    image: str,
    ssh_kwargs: dict,
    *,
    need_gpu: bool,
    mount_ssh: bool,
    config=None,
    cluster: ClusterDefinition | None = None,
    offline: bool = False,
) -> None:
    """Distribute the image and start the container everywhere, or raise.

    The image goes through the same distribution a launch uses: the cluster's
    transfer mode and interface, registered image-distribution providers (OCI
    Relay) and offline mode. It used to be pulled by every host from the
    registry directly, which ignored a ``delegated`` cluster and failed on an
    air-gapped one that already held the image on the control machine.
    """
    from sparkrun.orchestration.distribution import DistributionError, distribute_resources

    _say("Fetching %s on %d host(s) (first run transfers ~1 GB per host)...", image, len(host_list))
    if config is not None:
        op_config = _operation_config(config, cluster, ssh_kwargs)
        transfer_mode = (cluster.transfer_mode if cluster is not None else None) or "auto"
        try:
            _with_heartbeat(
                "image fetch",
                lambda: distribute_resources(
                    image,
                    "",
                    list(host_list),
                    str(op_config.hf_cache_dir),
                    op_config,
                    dry_run=False,
                    recipe_name="rdma-test",
                    transfer_mode=transfer_mode,
                    transfer_interface=cluster.transfer_interface if cluster is not None else None,
                    topology=cluster.topology if cluster is not None else None,
                    offline=offline,
                ),
            )
        except DistributionError as exc:
            hint = (
                "Offline: load it on the control machine or a host first (docker pull / docker load)."
                if offline
                else "Pull it manually or point --image at a reachable mirror."
            )
            raise RdmaTestError("Could not distribute %s: %s. %s" % (image, exc, hint)) from exc
    else:
        # A bare context has no config to scope distribution by: every host pulls for itself.
        from sparkrun.containers.sync import sync_image_to_hosts

        sync_failed = _with_heartbeat(
            "image fetch",
            lambda: sync_image_to_hosts(
                image,
                list(host_list),
                ssh_user=ssh_kwargs.get("ssh_user"),
                ssh_key=ssh_kwargs.get("ssh_key"),
                ssh_options=ssh_kwargs.get("ssh_options"),
                offline=offline,
            ),
        )
        if sync_failed:
            raise RdmaTestError(
                "Could not fetch %s on: %s. Pull it manually or point --image at a reachable mirror." % (image, ", ".join(sync_failed))
            )

    _say("Image ready on all %d host(s)", len(host_list))

    _say("Starting test container on %d host(s)...", len(host_list))
    start_failed, reasons = _start_containers(list(host_list), image, ssh_kwargs, need_gpu=need_gpu, mount_ssh=mount_ssh, dry_run=False)
    if start_failed:
        _stop_containers(list(host_list), ssh_kwargs, dry_run=False)
        # Quote docker's own message. Hosts normally fail identically (same
        # image, same script), so one shared cause is stated once rather than
        # repeated per host.
        distinct = {reasons.get(h, "") for h in start_failed}
        if len(distinct) == 1:
            detail = distinct.pop()
            body = "Could not start the RDMA test container on: %s" % ", ".join(start_failed)
            if detail:
                body += "\n  %s" % detail
        else:
            body = "Could not start the RDMA test container:\n" + "\n".join(
                "  %s: %s" % (h, reasons.get(h) or "unknown") for h in start_failed
            )
        raise RdmaTestError(body + "\nVerify docker is usable there: sparkrun setup check")
