"""Native-cluster distributed-init address selection."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

from sparkrun.utils.shell import quote

if TYPE_CHECKING:
    from sparkrun.runtimes._cluster_ops import ClusterContext

logger = logging.getLogger(__name__)

InitNetworkName = Literal["management", "ib"]


@dataclass(frozen=True, slots=True)
class InitNetworkCandidates:
    """Candidate address sets for native distributed initialization."""

    management_head_ip: str
    management_hosts: tuple[str, ...]
    ib_ip_map: Mapping[str, str]

    @classmethod
    def from_resolved(
        cls,
        management_head_ip: str,
        management_hosts: Sequence[str],
        ib_ip_map: Mapping[str, str],
    ) -> InitNetworkCandidates:
        return cls(
            management_head_ip=management_head_ip,
            management_hosts=tuple(management_hosts),
            ib_ip_map=ib_ip_map,
        )


@dataclass(frozen=True, slots=True)
class InitNetworkSelection:
    """Chosen address set for native distributed initialization."""

    head_ip: str
    hosts: tuple[str, ...]
    network: InitNetworkName


def select_init_network(ctx: ClusterContext, candidates: InitNetworkCandidates) -> InitNetworkSelection:
    """Choose management init addresses unless worker reachability requires IB.

    Native runtimes advertise one head address to every worker.  The
    management/default-route address remains the preferred path; CX7/IB
    addresses are substituted only when workers cannot reach that management
    head address and a complete, reachable IB address set is available.
    """
    management = InitNetworkSelection(
        head_ip=candidates.management_head_ip,
        hosts=candidates.management_hosts,
        network="management",
    )
    if ctx.dry_run or not ctx.worker_hosts:
        return management

    if workers_can_reach(ctx, candidates.management_head_ip):
        logger.info("  Init network: management (head=%s)", candidates.management_head_ip)
        return management

    ib_selection = _build_ib_selection(ctx, candidates)
    if ib_selection is None:
        logger.warning(
            "  Management init address %s is not reachable from all workers, "
            "but no complete IB/CX7 address map is available; keeping management init",
            candidates.management_head_ip,
        )
        return management

    if workers_can_reach(ctx, ib_selection.head_ip):
        logger.warning(
            "  Management init address %s is not reachable from all workers; using IB/CX7 init address %s",
            candidates.management_head_ip,
            ib_selection.head_ip,
        )
        logger.info("  Init network: ib (head=%s, hosts=%s)", ib_selection.head_ip, list(ib_selection.hosts))
        return ib_selection

    logger.warning(
        "  Management init address %s is not reachable from all workers, "
        "but IB/CX7 init address %s is also not reachable; keeping management init",
        candidates.management_head_ip,
        ib_selection.head_ip,
    )
    return management


def workers_can_reach(ctx: ClusterContext, target_ip: str) -> bool:
    """Return whether every worker host can reach *target_ip*."""
    if ctx.dry_run or not ctx.worker_hosts:
        return True

    from sparkrun.orchestration.ssh import resolve_parallel_cap

    with ThreadPoolExecutor(max_workers=resolve_parallel_cap(len(ctx.worker_hosts))) as pool:
        results = dict(pool.map(lambda host: _worker_reaches(ctx, host, target_ip), ctx.worker_hosts))

    failed = [host for host, ok in results.items() if not ok]
    if failed:
        logger.debug("  Init address %s failed reachability from workers: %s", target_ip, failed)
        return False
    return True


def _build_ib_selection(ctx: ClusterContext, candidates: InitNetworkCandidates) -> InitNetworkSelection | None:
    missing = [host for host in ctx.hosts if not candidates.ib_ip_map.get(host)]
    if missing:
        logger.debug("  IB/CX7 init candidates missing for host(s): %s", missing)
        return None
    return InitNetworkSelection(
        head_ip=candidates.ib_ip_map[ctx.head_host],
        hosts=tuple(candidates.ib_ip_map[host] for host in ctx.hosts),
        network="ib",
    )


def _worker_reaches(ctx: ClusterContext, worker_host: str, target_ip: str) -> tuple[str, bool]:
    from sparkrun.orchestration.primitives import run_command_on_host

    result = run_command_on_host(
        worker_host,
        _reachability_command(target_ip),
        ssh_kwargs=ctx.ssh_kwargs,
        timeout=6,
        dry_run=ctx.dry_run,
        quiet=True,
    )
    return worker_host, result.success


def _reachability_command(target_ip: str) -> str:
    target = quote(target_ip)
    return (
        """target=%s
if command -v ping >/dev/null 2>&1 && ping -c 1 -W 1 "$target" >/dev/null 2>&1; then
    exit 0
fi
if command -v nc >/dev/null 2>&1 && nc -z -w 1 "$target" 22 >/dev/null 2>&1; then
    exit 0
fi
exit 1
"""
        % target
    )
