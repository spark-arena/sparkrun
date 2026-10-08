"""Read-only setup probing shared by the readiness command and wizard."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import shlex
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.cluster_manager import ClusterDefinition

from sparkrun.core.setup_models import CheckContext, HostState

logger = logging.getLogger(__name__)


def resolve_setup_context(
    states: Mapping[str, HostState],
    *,
    config: SparkrunConfig,
    cluster: ClusterDefinition | None = None,
    cluster_name: str | None = None,
    strict: bool = True,
) -> CheckContext:
    from sparkrun.orchestration.executor import resolve_executor

    ctx = CheckContext(cluster_name=cluster_name, multi_host=len(states) > 1, config=config, strict=strict)
    for host, state in states.items():
        if state.hardware is None:
            continue
        try:
            executor = resolve_executor(cluster=cluster, config=config, host_hardware=state.hardware, rootless=False, auto_user=False)
            ctx.executor_names[host] = executor.executor_name
            ctx.gpu_access_modes[host] = executor.config.gpu_access_mode
            if executor.executor_name == "docker":
                ctx.ipc_exposure[host] = executor.config.ipc or ""
            elif executor.executor_name == "local":
                ctx.ipc_exposure[host] = "host"
        except Exception:
            logger.debug("Could not resolve setup executor for %s", host, exc_info=True)
    return ctx


def setup_discovery_script() -> str:
    """Hardware identification and minimal OS prerequisites, before optional probes."""
    from sparkrun.core.hardware_probe import generate_combined_probe_script
    from sparkrun.scripts import read_script

    return "(\n" + generate_combined_probe_script() + "\n)\n(\n" + read_script("setup_discover.sh") + "\n)\n"


def _discovered_state(host, result) -> HostState:
    from sparkrun.core.hardware_probe import _parse_probe_result
    from sparkrun.utils.text import parse_kv_output

    state = HostState(host=host)
    if not result.success or "CHECK_DISCOVERY_COMPLETE=1" not in result.stdout.splitlines():
        state.reachable = False
        state.error = result.stderr.strip()[:160] or "setup discovery did not complete"
        return state
    state.facts = parse_kv_output(result.stdout)
    try:
        if "SPARKRUN_PROBE_ACCEL_END" in result.stdout:
            state.hardware = _parse_probe_result(result.stdout)
    except Exception as exc:
        state.error = "Hardware identification failed: %s" % exc
    return state


def local_setup_step_supported(key: str, config) -> bool:
    """Qualify control-host discovery (e.g. fabric peer discovery) before probing."""
    import subprocess
    from sparkrun.orchestration.ssh import RemoteResult
    from sparkrun.core.setup_steps import setup_step_reason

    try:
        result = subprocess.run(["bash", "-s"], input=setup_discovery_script(), capture_output=True, text=True, timeout=30)
        state = _discovered_state("localhost", RemoteResult("localhost", result.returncode, result.stdout, result.stderr))
        context = resolve_setup_context({state.host: state}, config=config)
        # Peer discovery is evaluating whether a multi-host plan can be offered.
        context.multi_host = True
        return not setup_step_reason(key, state, context)
    except Exception:
        logger.debug("Local setup discovery failed", exc_info=True)
        return False


def probe_setup_hosts(
    hosts: Sequence[str],
    *,
    ssh_kwargs: dict[str, Any],
    config: SparkrunConfig,
    cluster: ClusterDefinition | None = None,
    cluster_name: str | None = None,
    discovery_only: bool = False,
    extra_mesh_peers: Sequence[str] = (),
) -> tuple[dict[str, HostState], CheckContext]:
    """Discover hardware, resolve owned plans, then probe only selected steps.

    ``discovery_only`` supports standalone action preflight without running
    readiness or topology probes before the requested step is approved.
    ``extra_mesh_peers`` are SSH-mesh members outside *hosts* (the wizard's
    control machine); every host's mesh probe also dials them, and they are
    kept on the returned context so reprobes measure the same mesh.
    """
    from sparkrun.core.setup_steps import setup_selection, setup_probe_script
    from sparkrun.orchestration.ssh import run_remote_script
    from sparkrun.scripts import read_script
    from sparkrun.utils.text import parse_kv_output

    discovery = setup_discovery_script()

    def discover(host):
        try:
            return _discovered_state(host, run_remote_script(host, discovery, timeout=60, quiet=True, **ssh_kwargs))
        except Exception as exc:
            return HostState(host=host, reachable=False, error="Setup discovery failed: %s" % exc)

    if not hosts:
        return {}, resolve_setup_context({}, config=config, cluster=cluster, cluster_name=cluster_name)
    with ThreadPoolExecutor(max_workers=min(len(hosts), 16)) as pool:
        states = dict(zip(hosts, pool.map(discover, hosts), strict=True))
        context = resolve_setup_context(states, config=config, cluster=cluster, cluster_name=cluster_name)
        context.extra_mesh_peers = tuple(dict.fromkeys(p for p in extra_mesh_peers if p not in hosts))
        if discovery_only:
            return states, context
        selected = {
            host: {key for key, reason in setup_selection(state, context)[1].items() if key != "hardware" and not reason}
            for host, state in states.items()
        }

        def readiness(host):
            state = states[host]
            if not selected[host]:
                return
            peers = " ".join([h for h in hosts if h != host and "ssh_mesh" in selected[h]] + list(context.extra_mesh_peers))
            script = read_script("setup_check.sh").format(steps=shlex.quote(" ".join(sorted(selected[host]))), peers=shlex.quote(peers))
            script = "(\n" + script + "\n)\n" + setup_probe_script(state, context)
            try:
                result = run_remote_script(host, script, timeout=60, quiet=True, **ssh_kwargs)
                if result.success and "CHECK_COMPLETE=1" in result.stdout.splitlines():
                    state.facts.update(parse_kv_output(result.stdout))
                    return
                state.error = result.stderr.strip()[:160] or "setup readiness probe did not complete"
            except Exception as exc:
                state.error = "Setup readiness probe failed: %s" % exc
            state.reachable = False

        list(pool.map(readiness, hosts))

    def candidates(key):
        return [h for h, state in states.items() if state.reachable and key in selected[h]]

    if targets := candidates("cx7"):
        from sparkrun.orchestration.networking import detect_cx7_for_hosts

        try:
            for host, detection in detect_cx7_for_hosts(targets, ssh_kwargs=ssh_kwargs).items():
                states[host].cx7 = detection
        except Exception:
            logger.debug("Setup CX7 probe failed", exc_info=True)
    if targets := candidates("rdma"):
        from sparkrun.api.setup._rdma import _run_probe

        try:
            for host, facts in _run_probe(targets, ssh_kwargs, dry_run=False).items():
                states[host].rdma = facts
        except Exception:
            logger.debug("Setup RDMA probe failed", exc_info=True)
    return states, context
